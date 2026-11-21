import json
import math

from fastapi.testclient import TestClient
import numpy as np
import pytest

from franka_compliant_control.client import CompliantClient
from franka_compliant_control.kinesthetic import collect
from franka_compliant_control.models import ControllerConfig
from franka_compliant_control.twin_model import rotation_distance


def http_client(stack_app):
    client = CompliantClient("http://testserver", timeout_s=5)
    client.http.close()
    client.http = stack_app
    return client


def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def test_push_moves_target_to_measured_pose_and_records(tmp_path):
    config = ControllerConfig(robot_model="fr3", translation_damping=[150]*3)
    from franka_compliant_control.service import create_app
    with TestClient(create_app(config=config, enable_sim_injection=True, automatic_recovery=False)) as http:
        client = http_client(http)
        pushed = []

        def report(event):
            # Push, then let go so the stop can confirm a stationary robot.
            if event["event"] == "sample" and len(pushed) < 2 and event["elapsed_s"] > (.2, 1.2)[len(pushed)]:
                wrench = [9, 0, 0, 0, 0, 0] if not pushed else [0]*6
                http.post("/sim/wrench", json={"wrench": wrench}).raise_for_status()
                pushed.append(True)

        summary = collect(client, tmp_path/"run", translation_threshold_m=.001,
                          rotation_threshold_rad=math.radians(5), duration_s=2.,
                          expected_backend="sim", report=report)
        assert http.get("/state").json()["running"] is False
    assert summary["stop_confirmed"] and summary["error"] is None and summary["reason"] == "duration"
    assert summary["retargets"] > 0 and summary["samples"] > 50
    trace, actions = rows(tmp_path/"run/trace.jsonl"), rows(tmp_path/"run/actions.jsonl")
    assert len(actions) == len(trace)+1  # startup target, then one per tick
    first = trace[0]["state"]
    assert actions[0]["action"]["position"] == json.loads((tmp_path/"run/manifest.json").read_text())["initial_state"]["position"]
    for row, action in zip(trace, actions[1:]):
        # Target is never more than the threshold from the measured pose unless it is reset to it.
        if row["retarget"]:
            assert action["action"]["position"] == row["state"]["position"]
            assert row["target_distance_m"] > .001
        else:
            assert action["action"]["position"] == row["target_position"]
            assert row["target_distance_m"] <= .001
    moved = np.linalg.norm(np.subtract(trace[-1]["state"]["position"], first["position"]))
    assert moved > .002  # Target followed the push well beyond one threshold.
    box = summary["workspace"]["tcp_position_m"]
    assert box["max"][0]-box["min"][0] > .002
    assert rotation_distance(first["quaternion_xyzw"], actions[0]["action"]["quaternion_xyzw"]) < 1e-3


def test_refuses_automatic_recovery_without_writing(tmp_path):
    from franka_compliant_control.service import create_app
    with TestClient(create_app(config=ControllerConfig(robot_model="fr3"), automatic_recovery=True)) as http:
        with pytest.raises(RuntimeError, match="automatic recovery"):
            collect(http_client(http), tmp_path/"run", expected_backend="sim")
        assert http.get("/state").json()["running"] is False
    assert not (tmp_path/"run").exists()
