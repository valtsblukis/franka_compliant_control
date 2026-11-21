"""Native capture API uses simulated backend only; never a robot connection."""

import io
import json
import time
import numpy as np
import pytest
from fastapi.testclient import TestClient
from franka_compliant_control import _native
from franka_compliant_control.models import ControllerConfig
from franka_compliant_control.service import create_app
from franka_compliant_control.native_telemetry import COLUMNS, quality

pytestmark = pytest.mark.skipif(
    not hasattr(_native.Backend, "prepare_telemetry"),
    reason="instrumented native build required",
)


def test_native_experiment_speed_guard_stops_between_http_polls():
    with TestClient(create_app(config=ControllerConfig(),backend_kind='sim')) as c:
        s=c.get('/state').json();assert c.post('/telemetry/prepare',json={'capacity':5000,'joint_speed':.00001}).status_code==200
        goal=list(s['position']);goal[0]+=.01
        assert c.post('/target_ee_pose',json={'position':goal,'quaternion_xyzw':s['quaternion_xyzw'],'seq':1,'valid_for_s':1.}).status_code==202
        deadline=time.monotonic()+3
        while c.get('/state').json()['running'] and time.monotonic()<deadline:time.sleep(.01)
        assert not c.get('/state').json()['running']
        status=c.get('/telemetry/status').json();assert status['speed_guard_trip']==1
        payload=c.get('/telemetry/data');assert payload.status_code==200
        with np.load(io.BytesIO(payload.content),allow_pickle=False) as a:assert np.any((a['ids'][:,6]&512)!=0)
        assert c.post('/telemetry/clear',json={'capture_id':status['capture_id']}).status_code==200


def test_capture_lifecycle_overflow_and_coherent_snapshot(tmp_path):
    with TestClient(create_app(config=ControllerConfig(), backend_kind="sim")) as c:
        state = c.get("/state").json()
        assert (
            abs(
                state["sample_age_s"]
                - (
                    state["state_timestamp_monotonic_s"]
                    - state["sample_timestamp_monotonic_s"]
                )
            )
            < 1e-8
        )
        assert c.post("/telemetry/prepare", json={"capacity": True}).status_code == 422
        assert c.post("/telemetry/prepare", json={"capacity": 100}).status_code == 200
        assert c.post("/telemetry/prepare", json={"capacity": 100}).status_code == 409
        pose = {
            "position": state["position"],
            "quaternion_xyzw": state["quaternion_xyzw"],
            "seq": 1,
            "valid_for_s": 1.0,
        }
        assert c.post("/target_ee_pose", json=pose).status_code == 202
        time.sleep(0.18)
        assert c.get("/telemetry/data").status_code == 409
        assert c.post("/telemetry/prepare", json={"capacity": 100}).status_code == 409
        assert c.post("/stop?wait_s=2.5").json()["stopped"]
        status = c.get("/telemetry/status").json()
        assert status["count"] == 100 and status["dropped"] > 0 and not status["active"]
        assert (
            c.post(
                "/telemetry/clear", json={"capture_id": status["capture_id"]}
            ).status_code
            == 409
        )
        payload = c.get("/telemetry/data")
        assert payload.status_code == 200
        path = tmp_path / "capture.npz"
        path.write_bytes(payload.content)
        result = quality(path)
        assert (
            not result["recording_is_complete"] and result["host_timestamp_order_valid"]
        )
        with np.load(io.BytesIO(payload.content), allow_pickle=False) as a:
            assert a["values"].shape == (100, len(COLUMNS))
            assert a["ids"].dtype == np.uint64
            assert np.array_equal(a["ids"][:, 0], np.arange(100))
        assert (
            c.post(
                "/telemetry/clear", json={"capture_id": status["capture_id"] + 1}
            ).status_code
            == 409
        )
        assert (
            c.post(
                "/telemetry/clear", json={"capture_id": status["capture_id"]}
            ).status_code
            == 200
        )
        assert c.post("/telemetry/prepare", json={"capacity": 1000}).status_code == 200
        assert c.get("/health").json()["phase"] == "disarmed"
