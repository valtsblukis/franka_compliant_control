from franka_compliant_control.client import CompliantClient


def test_client_has_only_setpoint_start_and_no_arm_or_session_api(monkeypatch):
    client = CompliantClient()
    calls = []
    def request(method, path, **kwargs):
        calls.append((path, kwargs))
        return {"accepted": True}
    monkeypatch.setattr(client, "_request", request)
    assert not hasattr(client, "arm") and not hasattr(client, "session_id")
    client.send_pose([.4, 0, .3], seq=0)
    assert calls[-1][1]["json"] == {"position": [.4, 0, .3], "seq": 0}
    client.send_joint_state([0]*7, valid_for_s=.1)
    second = calls[-1][1]["json"]
    assert second["seq"] == 1 and second["valid_for_s"] == .1
    assert "session_id" not in second and "deadline_s" not in second
    client.stop()
    client.send_pose([.4, 0, .3])
    assert calls[-1][1]["json"]["seq"] == 1
    client.close()
    assert calls[-1][0] == "/stop"
    assert all(path != "/arm" for path, _ in calls)
