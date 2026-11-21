from copy import deepcopy
import pytest
from fastapi.testclient import TestClient

from franka_compliant_control.client import CompliantClient
from franka_compliant_control.commissioning import hold_current_pose
from franka_compliant_control.models import ControllerConfig
from franka_compliant_control.service import create_app


def test_bounded_hold_through_http_simulator_uses_post_arm_pose():
    events = []
    with TestClient(create_app(config=ControllerConfig(robot_model="fr3"))) as http:
        client = CompliantClient("http://testserver", timeout_s=5)
        client.http.close()
        client.http = http
        summary = hold_current_pose(client, events.append, expected_backend="sim")
        assert summary["stopped"] and summary["samples"] >= 90
        assert events[-1]["result"]["state"]["phase"] == "disarmed"
        assert events[-1]["result"]["state"]["applied_seq"] > 0
        assert summary["max_position_drift_m"] < .001


@pytest.mark.parametrize("yielding", [False, True])
def test_hold_uses_fresh_arm_pose_and_stops_on_drift_or_yielding(yielding):
    class Client:
        sent = []
        armed = False
        stopped = False
        def state(self):
            return {"backend": "robot", "phase": "active" if self.armed else "disarmed",
                    "running": self.armed, "read_only": False, "sample_valid": True,
                    "sample_age_s": 0, "position": ([.5 if yielding else .506,0,.4] if self.armed else [.4,0,.4]),
                    "yielding": yielding,
                    "quaternion_xyzw": [0,0,0,1]}
        def arm(self):
            self.armed = True
            initial = deepcopy(self.state())
            initial["position"] = [.5,0,.4]
            return {"state": initial}
        def send_pose(self, position, quaternion):
            self.sent.append((position, quaternion))
        def stop(self, wait_s):
            self.stopped = True
            return {"stopped": True}
    client = Client()
    with pytest.raises(RuntimeError, match="yielded" if yielding else "Hold drift"):
        hold_current_pose(client, lambda event: None)
    assert client.sent == [([.5,0,.4], [0,0,0,1])]
    assert client.stopped
