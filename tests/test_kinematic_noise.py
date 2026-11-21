import numpy as np
import pytest
from franka_compliant_control.simulation.kinematic_noise import KinematicNoiseModel


def config():
    return dict(
        schema="fcc-kinematic-noise-v1",
        period_s=0.001,
        scale=[1e-6] * 7 + [0.001] * 7,
        A=(np.eye(14) * 0.5).tolist(),
        innovation_covariance=(np.eye(14) * 0.75).tolist(),
        stationary_covariance=np.eye(14).tolist(),
    )


def test_stationary_variance_replay_and_period():
    m = KinematicNoiseModel(config())
    m.reset(2)
    a = np.array(
        [np.r_[*m.observe(np.zeros(7), np.zeros(7), 0.001)] for _ in range(10000)]
    )
    assert np.allclose(np.std(a, axis=0), m.scale, rtol=0.04)
    m.reset(2)
    assert np.array_equal(np.r_[*m.observe(np.zeros(7), np.zeros(7), 0.001)], a[0])
    with pytest.raises(ValueError, match="period"):
        m.observe(np.zeros(7), np.zeros(7), 0.002)
    c = config()
    c["A"] = np.eye(14).tolist()
    with pytest.raises(ValueError, match="stable"):
        KinematicNoiseModel(c)


def test_native_measurement_separate_from_physics():
    pytest.importorskip("mujoco")
    from twin_helpers import home_fixture
    from franka_compliant_control.simulation.world import SimulationWorld
    from franka_compliant_control.simulation.native import NativeControllerSimulation
    from franka_compliant_control.twin_model import PlantParameters

    h = home_fixture()
    world = SimulationWorld(h, PlantParameters())
    sim = NativeControllerSimulation(
        world, kinematic_sensor=KinematicNoiseModel(config())
    )
    q = world.q.copy()
    assert not np.array_equal(sim.sample.q, q)
    assert np.array_equal(world.q, q)
    assert np.allclose(sim.twist, sim.sample.jacobian @ sim.sample.dq)
    assert (
        sim.manifest()["kinematic_sensor"]["config"]["schema"]
        == "fcc-kinematic-noise-v1"
    )
