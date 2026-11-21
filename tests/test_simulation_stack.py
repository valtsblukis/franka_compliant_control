"""Reusable simulation layers: no robot device or HTTP calls."""

from dataclasses import asdict
from pathlib import Path
import json
import xml.etree.ElementTree as ET
import numpy as np
import pytest

pytest.importorskip("mujoco")
import mujoco
from twin_helpers import home_fixture
from franka_compliant_control.twin_model import PlantParameters
from franka_compliant_control.twin_experiment import ManualPosePolicy
from franka_compliant_control.mujoco_twin import simulation_trace
from franka_compliant_control.simulation.world import SimulationWorld
from franka_compliant_control.simulation.native import NativeControllerSimulation
from franka_compliant_control.simulation.geometry import (
    RobotCollisionGeometry,
    CollisionGeometryConfig,
)
from franka_compliant_control.simulation.components import SceneComponent
from franka_compliant_control.simulation.contacts import (
    Contact,
    ContactReport,
    ContactAccumulator,
)
from franka_compliant_control.simulation.obstacles import (
    DynamicObstacles,
    MovableObstacleSpec,
    ObstacleSchedule,
    randomized_obstacles,
)
from franka_compliant_control.simulation.sensors import (
    FilteredContactSensor,
    ContactSensorConfig,
    FrankaWrenchSurrogate,
)
from franka_compliant_control.simulation.disturbances import (
    JointTorqueDisturbances,
    TorquePulse,
)


@pytest.fixture(scope="module")
def home():
    return home_fixture()


def scene(home, *, friction=0.0, lifetime=False):
    obstacle = DynamicObstacles(
        [
            MovableObstacleSpec(
                "test", mass_kg=0.8, capsule_length_m=0, support_stiffness_N_m=12
            )
        ],
        ObstacleSchedule(enabled=False),
    )
    world = SimulationWorld(
        home,
        PlantParameters(coulomb=(friction,) * 7),
        components=(RobotCollisionGeometry(), obstacle),
    )
    sensor = FilteredContactSensor(
        ContactSensorConfig(noise_std=(0.0,) * 6, time_constant_s=0.02)
    )
    return NativeControllerSimulation(world, sensor=sensor), obstacle


def test_lower_layers_do_not_import_task_training_or_device_service():
    import subprocess, sys

    code = """import sys
from franka_compliant_control.simulation.world import SimulationWorld
assert 'franka_compliant_control._native' not in sys.modules
from franka_compliant_control.simulation.native import NativeControllerSimulation
for module in ['gymnasium','torch','scipy','franka_compliant_control.service','franka_compliant_control.mujoco_twin']:
    assert module not in sys.modules, module
"""
    subprocess.run([sys.executable, "-c", code], check=True)


def test_free_core_matches_frozen_legacy_control_plant_trace(home):
    params = PlantParameters()
    policy = ManualPosePolicy.from_home(home, offset_m=0.01, move_s=1.0, dwell_s=0.3)
    core = NativeControllerSimulation(SimulationWorld(home, params))
    for tick, row in enumerate(simulation_trace(home, policy, params)):
        if tick >= 700:
            break
        assert core.measurement().positions == pytest.approx(
            row["positions"], abs=2e-11
        )
        assert core.measurement().velocities == pytest.approx(
            row["velocities"], abs=2e-10
        )
        if tick % 20 == 0:
            target = policy.sample(tick * 0.001)
            core.submit_pose(target["position"], target["quaternion_xyzw"])
        core.step_control()


def test_optional_inactive_bodies_do_not_change_robot_dynamics(home):
    plain = NativeControllerSimulation(SimulationWorld(home))
    extra, obstacle = scene(home, friction=0.2)
    # Use exactly matching robot plant and no sensor effect on saturated control.
    goal = np.array(home["home"]["position"]) + [0, 0.005, 0]
    for tick in range(250):
        if tick % 20 == 0:
            for sim in [plain, extra]:
                sim.submit_pose(goal, home["home"]["quaternion_xyzw"])
        plain.step_control()
        extra.step_control()
        assert plain.world.q == pytest.approx(extra.world.q, abs=2e-10)
    assert extra.world.model.nq == 14 and extra.world.model.nv == 13
    assert not extra.world.last_contacts.contacts


def test_robot_addressing_is_not_full_generalized_state(home):
    class PrefixBody(SceneComponent):
        def build(self, root, home):
            body = ET.Element("body", name="before_robot", pos="20 0 20")
            ET.SubElement(body, "freejoint")
            ET.SubElement(
                body, "inertial", pos="0 0 0", mass="1", diaginertia=".1 .1 .1"
            )
            root.find("worldbody").insert(0, body)

    world = SimulationWorld(home, components=(PrefixBody(),))
    assert world.robot_qpos.tolist() == list(range(7, 14))
    assert world.robot_dofs.tolist() == list(range(6, 13))
    model = world.plant.model_quantities()
    assert model["mass"].shape == (7, 7) and model["jacobian"].shape == (6, 7)
    sim = NativeControllerSimulation(world)
    sim.submit_pose(home["home"]["position"], home["home"]["quaternion_xyzw"])
    for _ in range(100):
        sim.step_control()
    assert max(abs(world.q - home["home"]["joints"])) < 1e-5


def test_scene_instances_are_not_shared(home):
    component = RobotCollisionGeometry()
    SimulationWorld(home, components=(component,))
    with pytest.raises(ValueError, match="shared"):
        SimulationWorld(home, components=(component,))


def test_expired_delayed_command_does_not_extend_watchdog(home):
    sim = NativeControllerSimulation(SimulationWorld(home))
    sim.submit_pose(
        home["home"]["position"],
        home["home"]["quaternion_xyzw"],
        valid_for_s=0.005,
        delay_steps=10,
    )
    for _ in range(250):
        step = sim.step_control()
        if step.phase != "active":
            break
    assert step.reason == "command_timeout" and sim.now == pytest.approx(0.2)
    assert any(e["kind"] == "command_rejected" for e in sim.world.events)


def test_spawn_rejects_intersection_without_mutating_robot_or_scene(home):
    sim, obstacle = scene(home)
    world = sim.world
    q = world.data.qpos.copy()
    v = world.data.qvel.copy()
    assert not obstacle.spawn(world, 0, home["home"]["position"])
    assert np.array_equal(world.data.qpos, q) and np.array_equal(world.data.qvel, v)
    assert obstacle.slots[0]["status"] == "inactive" and not world.events
    assert obstacle.spawn(world, 0, np.array(home["home"]["position"]) + [0, 0.11, 0])
    with pytest.raises(ValueError, match="teleport"):
        obstacle.spawn(world, 0, home["home"]["position"])


def test_contact_force_frame_projection_and_movable_dynamics(home):
    sim, obstacle = scene(home)
    world = sim.world
    origin = np.array(home["home"]["position"])
    initial = origin + [0, 0.11, 0]
    assert obstacle.spawn(world, 0, initial)
    accumulator = ContactAccumulator()
    checked = False
    displacement = 0.0
    for tick in range(2400):
        if tick % 20 == 0:
            sim.submit_pose(
                origin + [0, 0.05 * tick * 0.001, 0], home["home"]["quaternion_xyzw"]
            )
        result = sim.step_control()
        for dt, contacts in result.physics_contacts:
            accumulator.add(contacts, dt)
            if contacts.peak_body_force_N > 0.1 and not checked:
                # Zero frictionloss, no active joint limit: all constraint force
                # on robot is contact. This checks sign, frame, and actual link J.
                assert -np.array(contacts.robot_on_environment_torque) == pytest.approx(
                    world.data.qfrc_constraint[world.robot_dofs], abs=1e-7
                )
                object_force = np.zeros(3)
                for c in contacts.contacts:
                    if c.body2 == "obstacle_test":
                        object_force += c.force_on_body2
                    if c.body1 == "obstacle_test":
                        object_force -= np.array(c.force_on_body2)
                d = obstacle.slots[0]["dof"]
                assert object_force == pytest.approx(
                    world.data.qfrc_constraint[d : d + 3], abs=1e-7
                )
                assert object_force[1] > 0
                checked = True
        displacement = max(
            displacement, float(np.linalg.norm(world.data.qpos[7:10] - initial))
        )
    assert checked and accumulator.peak_force_N > 0.1 and accumulator.impulse_Ns > 0
    assert displacement > 0.02  # A finite body is actually pushed, not a virtual wall.
    assert world.model.body_mass[obstacle.slots[0]["body"]] == pytest.approx(0.8)
    assert obstacle.request_despawn(world, 0)
    assert obstacle.slots[0]["status"] == "withdrawing"
    for tick in range(4000):
        if tick % 20 == 0:
            sim.submit_pose(sim.sample.position, sim.sample.quaternion_xyzw)
        sim.step_control()
        if obstacle.slots[0]["status"] == "inactive":
            break
    assert obstacle.slots[0]["status"] == "inactive"
    assert (
        world.events[-1]["kind"] == "obstacle_deactivated"
        and world.events[-1]["unloaded"]
    )
    assert (
        accumulator.peak_force_N > 0.1
    )  # Removing geometry cannot erase recorded cost.


def test_withdrawal_does_not_overshoot_into_static_scene(home):
    origin=np.array(home['home']['position'])+[0,.4,0]
    class Block(SceneComponent):
        def build(self,root,home):
            b=ET.SubElement(root.find('worldbody'),'body',name='withdrawal_stop',pos=' '.join(map(str,origin+[0,.31,0])))
            ET.SubElement(b,'geom',type='box',size='.15 .005 .15',contype='2',conaffinity='2',solref='.04 1')
    objects=DynamicObstacles([MovableObstacleSpec('withdraw',mass_kg=1.,radius_m=.03,capsule_length_m=0.,support_stiffness_N_m=25.,support_damping_Ns_m=4.)],ObstacleSchedule(enabled=False))
    world=SimulationWorld(home,components=(RobotCollisionGeometry(),Block(),objects))
    sim=NativeControllerSimulation(world)
    assert objects.spawn(world,0,origin)
    assert objects.request_despawn(world,0)
    peak=0.
    for i in range(5000):
        if i%20==0:sim.submit_pose(home['home']['position'],home['home']['quaternion_xyzw'])
        result=sim.step_control()
        peak=max(peak,max((r.peak_body_force_N for _,r in result.physics_contacts),default=0.))
        if objects.slots[0]['status']=='inactive':break
    assert objects.slots[0]['status']=='inactive'
    assert peak<1e-6
    assert any(e.get('swept_clearance_checked') for e in world.events)


def contact(force, body="object"):
    return Contact(
        "robot",
        body,
        0,
        1,
        (0.0, 0.0, 0.0),
        tuple(force),
        (0.0, 0.0, 0.0),
        abs(force[0]),
        0.001,
        True,
        False,
        True,
        True,
    )


def test_opposing_contact_forces_cannot_cancel_reward_cost():
    report = ContactReport((contact([10, 0, 0]), contact([-10, 0, 0])))
    assert report.loads_by_body["object"] == 20
    stats = ContactAccumulator(4.0)
    stats.add(report, 0.001)
    assert stats.impulse_Ns == pytest.approx(0.02)
    assert stats.excess_force_squared_N2s == pytest.approx(0.256)


def test_contact_integrals_are_physics_timestep_consistent():
    report = ContactReport((contact([8, 0, 0]),))
    a = ContactAccumulator()
    b = ContactAccumulator()
    for _ in range(20):
        a.add(report, 0.001)
    for _ in range(40):
        b.add(report, 0.0005)
    assert a.impulse_Ns == pytest.approx(b.impulse_Ns)
    assert a.excess_force_squared_N2s == pytest.approx(0.32)


def test_external_disturbance_is_not_attenuated_as_free_resistance(home):
    from franka_compliant_control.simulation.presets import wrench_observer_config
    artifact = wrench_observer_config()
    from franka_compliant_control.simulation.sdk import offline_model

    sample = offline_model(home).sample(
        home["home"]["joints"], np.zeros(7), np.zeros(7), np.zeros(6)
    )
    artifact["resistance_gain"] = [0.0] * 7
    artifact["joint_offset"] = [0.0] * 7
    artifact["joint_position_slope"] = [0.0] * 7
    artifact["time_constant_s"] = 0.0
    sensor = FrankaWrenchSurrogate(artifact)
    external = np.array([0.1, -0.2, 0.1, 0, 0, 0, 0])
    reading = sensor.observe(sample, np.full(7, 100.0), external, 0.001)
    assert reading.joint_torques == pytest.approx(external)


def test_joint_disturbance_reaches_sensor_as_external_not_free_resistance(home):
    pulse = TorquePulse(0.02, 0.1, (0, 0.15, 0, 0, 0, 0, 0))
    world = SimulationWorld(home, components=(JointTorqueDisturbances([pulse]),))
    sensor = FilteredContactSensor(
        ContactSensorConfig(
            time_constant_s=0.0, noise_std=(0.0,) * 6, free_residual_gain=0.0
        )
    )
    sim = NativeControllerSimulation(world, sensor=sensor)
    for i in range(50):
        if i % 20 == 0:
            sim.submit_pose(home["home"]["position"], home["home"]["quaternion_xyzw"])
        sim.step_control()
    expected = (
        -np.asarray(pulse.torque_Nm)
        * np.sin(np.pi * (sim.now - pulse.start_s) / pulse.duration_s) ** 2
    )
    assert sim.sample.external_torque == pytest.approx(expected, abs=1e-12)


def test_random_obstacles_are_reproducible_and_finite_mass():
    a = randomized_obstacles(np.random.default_rng(91))
    b = randomized_obstacles(np.random.default_rng(91))
    assert a.manifest() == b.manifest()
    assert all(s.mass_kg > 0 and np.all(s.inertia() > 0) for s in a.specs)
