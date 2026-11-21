"""Exercise the actual foreground CLI using only the native simulation plant."""
import json
import math
from pathlib import Path
import signal
import shutil
import subprocess
import sys
import time
import weakref

import pytest

from franka_compliant_control import _native
from franka_compliant_control.local_client import LocalControllerClient
from franka_compliant_control.models import ControllerConfig


def test_close_releases_connection_even_when_caller_retains_service(monkeypatch):
    closed = []
    class Connection:
        def __init__(self, *_):
            pass

        def close(self):
            closed.append(True)

    monkeypatch.setattr(_native, "Backend", Connection)
    client = LocalControllerClient(ControllerConfig(), backend_kind="sim")
    retained_service = client.controller  # As an exception traceback might do.
    connection = weakref.ref(retained_service.backend)
    client.close()
    assert closed == [True] and connection() is None
    client.close()  # Idempotent cleanup.


@pytest.mark.parametrize("stop_signal", [signal.SIGINT, signal.SIGTERM, signal.SIGHUP])
@pytest.mark.parametrize("script", ["compliant_hold", "impedance_hold"])
def test_foreground_signal_stops_and_exits_without_children(tmp_path, stop_signal, script):
    log = tmp_path / "hold.jsonl"
    process = subprocess.Popen(
        [sys.executable, f"scripts/{script}.py", "--backend", "sim",
         "--duration", "0", "--output", str(log)],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    try:
        deadline = time.monotonic() + 8
        while time.monotonic() < deadline:
            assert process.poll() is None, process.communicate()
            if log.exists() and '"event": "sample"' in log.read_text():
                break
            time.sleep(.02)
        else:
            pytest.fail("foreground hold did not begin streaming")
        # The native control worker is a thread, not an independent service.
        assert not Path(f"/proc/{process.pid}/task/{process.pid}/children").read_text().strip()
        process.send_signal(stop_signal)
        stdout, stderr = process.communicate(timeout=6)
        assert process.returncode == 0, (stdout, stderr)
        assert "Stopped and disarmed." in stdout
        assert "leaves no service running" in stdout
        events = [json.loads(line) for line in log.read_text().splitlines()]
        stop = next(e for e in events if e["event"] == "stop")
        assert stop["result"]["stopped"]
        summary = next(e for e in events if e["event"] == "summary")
        assert summary["reason"] == "interrupted"
    finally:
        if process.poll() is None:
            process.kill()
        process.communicate(timeout=3)


@pytest.mark.parametrize(("script", "profile"), [
    ("compliant_hold", None), ("compliant_hold", "commissioning"),
    ("compliant_hold", "perturbation"), ("impedance_hold", None),
    ("impedance_hold", "contact"), ("impedance_hold", "impedance"),
])
def test_foreground_duration_closes_controller(tmp_path, profile, script):
    log = tmp_path / "hold.jsonl"
    profile_args = ["--profile", profile] if profile else []
    expected_profile = (profile or "contact") if script == "impedance_hold" else (profile or "perturbation")
    continuous = script == "impedance_hold"
    script_path = Path(f"scripts/{script}.py")
    if profile == "commissioning":
        # Exercise the local-profile lookup without depending on a lab's
        # ignored configuration or writing into the source checkout.
        fixture_root = tmp_path / "commissioning"
        (fixture_root / "scripts").mkdir(parents=True)
        shutil.copy2("configs/controller.fr3.example.json", fixture_root / "local-controller.json")
        script_path = Path(shutil.copy2(script_path, fixture_root / "scripts" / script_path.name))
    result = subprocess.run(
        [sys.executable, str(script_path), "--backend", "sim",
         *profile_args, "--duration", ".25", "--output", str(log)],
        capture_output=True, text=True, timeout=8)
    assert result.returncode == 0, (result.stdout, result.stderr)
    assert "Stopped and disarmed." in result.stdout
    assert "Compliance is inactive." in result.stdout
    events = [json.loads(line) for line in log.read_text().splitlines()]
    summary = next(e for e in events if e["event"] == "summary")
    assert summary["reason"] == "duration" and summary["samples"] > 0
    before = next(e for e in events if e["event"] == "before")
    assert before["max_displacement_m"] == (None if continuous else
                                            (.05 if expected_profile == "commissioning" else .15))
    assert before["max_rotation_rad"] == (None if continuous else
                                         pytest.approx(math.radians(15 if expected_profile == "commissioning" else 30)))
    assert f"Profile: {expected_profile}" in result.stdout
    launch = next(e for e in events if e["event"] == "launch")
    assert launch["profile"] == expected_profile
    config = launch["controller_config"]
    assert config["translation_stiffness"] == [400]*3
    expected_mode = "continuous_impedance" if continuous else "contact_yielding"
    assert config["compliance_mode"] == expected_mode
    assert before["state"]["compliance_mode"] == expected_mode
    assert f"Control law: {expected_mode}" in result.stdout
    assert config["translation_velocity"] == (.02 if expected_profile == "commissioning" else .2)
    assert config["contact_moment_stop"] == (20 if expected_profile == "contact" else
                                             (2 if expected_profile == "commissioning" else 10))
    assert config["bounded_cartesian_spring"] == (expected_profile == "contact")
    if expected_profile != "commissioning":
        assert config["measured_joint_velocity_limits"][5] == (4.18 if continuous else 1.3)
        speeds = "2.62, 2.62, 2.62, 2.62, 5.26, 4.18, 5.26" if continuous else "1.5, 0.9, 1.2, 1.1, 2, 1.3, 2"
        assert f"joints 1-7: {speeds} rad/s" in result.stdout
    if continuous:
        assert "IMPEDANCE" in result.stdout and "YIELDING" not in result.stdout
        assert "contact-limit" in result.stdout
