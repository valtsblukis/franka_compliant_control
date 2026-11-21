"""Simulation-only example: replace the progressive equilibrium without rearming.

These are direct goal changes, not paced motion waypoints. Replace the goal
selection in an application with planner/operator targets, while continuing the
50 Hz stream even when the robot is blocked. No robot connection is made here.
"""
from pathlib import Path
import time

from franka_compliant_control.local_client import LocalControllerClient
from franka_compliant_control.models import ControllerConfig


def main():
    root = Path(__file__).resolve().parents[1]
    config = ControllerConfig.model_validate_json((root/"configs/controller.fr3.osc.progressive.json").read_text())
    with LocalControllerClient(config, backend_kind="sim") as robot:
        try:
            initial = robot.arm()["state"]
            home, orientation = initial["position"], initial["quaternion_xyzw"]
            for offset in ([.2,0,0], [0,.2,0], [-.2,0,0], [0,0,0]):
                goal = [p+d for p, d in zip(home, offset)]
                print(f"New equilibrium: {goal}; no rearm or wait-for-reached", flush=True)
                end = time.monotonic()+2
                while time.monotonic() < end:
                    robot.send_pose(goal, orientation)
                    state = robot.state()
                    if not state["running"] or state["phase"] not in ("armed", "active"):
                        raise RuntimeError(f"Simulation controller stopped: {state['reason']}")
                    if not state["sample_valid"] or state["sample_age_s"] > .05:
                        raise RuntimeError("Invalid/stale simulation state")
                    time.sleep(.02)
                state = robot.state()
                print(f"Applied sequence {state['applied_seq']}; measured position {state['position']}; "
                      f"spring wrench {state['spring_wrench']}", flush=True)
        finally:
            stopped = robot.stop(wait_s=2.5)
            if not stopped["stopped"]:
                raise RuntimeError(f"Simulation stop unconfirmed: {stopped['state']['reason']}")
    print("Stopped and disarmed. Simulation connection closed.")


if __name__ == "__main__":
    main()
