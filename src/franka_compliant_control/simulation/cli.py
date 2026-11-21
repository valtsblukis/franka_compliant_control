"""Offline hold smoke test for the model and optional production-controller layer."""
import argparse
import json
from pathlib import Path
import numpy as np
from .presets import reference_context, identified_plant, nearstatic_noise_config
from .world import SimulationWorld


def smoke(*, controller=False, seconds=1.0, noise=False):
    if not np.isfinite(seconds) or not .01 <= seconds <= 60:
        raise ValueError('Use a finite duration in [.01, 60] seconds')
    if noise and not controller:
        raise ValueError('Kinematic sensing needs the controller layer')
    world = SimulationWorld(reference_context(), identified_plant())
    initial = world.q.copy()
    if controller:
        from .native import NativeControllerSimulation
        from .kinematic_noise import KinematicNoiseModel
        sim = NativeControllerSimulation(world, kinematic_sensor=KinematicNoiseModel(nearstatic_noise_config()) if noise else None)
        p, q = sim.sample.position.copy(), sim.sample.quaternion_xyzw.copy()
        for tick in range(round(seconds/sim.control_dt)):
            if tick % 20 == 0:
                sim.submit_pose(p, q, valid_for_s=.2)
            result = sim.step_control()
            if result.phase != 'active':
                raise RuntimeError(result.reason)
        manifest = sim.manifest()
    else:
        for _ in range(round(seconds/world.physics_dt)):
            world.step_physics(world.plant.model_quantities()['gravity'])
        manifest = world.manifest()
    change = float(np.max(abs(world.q-initial)))
    if not np.isfinite(change) or change > .01:
        raise RuntimeError('Hold smoke failed')
    return dict(passed=True, hardware_IO=False, maximum_joint_change_rad=change, manifest=manifest)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--controller', action='store_true')
    p.add_argument('--noise', action='store_true')
    p.add_argument('--seconds', type=float, default=1.)
    p.add_argument('--output', type=Path)
    a = p.parse_args(argv)
    result = smoke(controller=a.controller, seconds=a.seconds, noise=a.noise)
    if a.output:
        with a.output.open('x') as stream:
            json.dump(result, stream, indent=2); stream.write('\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
