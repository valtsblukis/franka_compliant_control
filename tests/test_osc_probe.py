import importlib.util
import json
import math
from pathlib import Path
import subprocess
import sys

import pytest

spec = importlib.util.spec_from_file_location("osc_orientation_probe", "scripts/osc_orientation_probe.py")
probe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(probe)


def test_base_rotation_composition_and_distance():
    # A 180-degree tool flip must not reverse the base-frame increment.
    q = [1, 0, 0, 0]
    angle = math.radians(2)
    r = [0, 0, math.sin(angle/2), math.cos(angle/2)]
    target = probe.quaternion_multiply(r, q)
    assert target == pytest.approx([math.cos(angle/2), math.sin(angle/2), 0, 0])
    assert probe.angular_distance(target, q) == pytest.approx(angle)


def test_probe_hardware_requires_execute():
    result = subprocess.run([sys.executable, "scripts/osc_orientation_probe.py", "--backend", "robot"],
                            capture_output=True, text=True, timeout=5)
    assert result.returncode == 2 and "Hardware requires --execute" in result.stderr


def test_probe_sequence_and_stop_in_simulation(tmp_path):
    log = tmp_path/"probe.jsonl"
    result = subprocess.run([sys.executable, "scripts/osc_orientation_probe.py", "--axis", "z",
                             "--degrees", "1", "--target-time", "1.5", "--return-time", "1.5",
                             "--output", str(log)], capture_output=True, text=True, timeout=12)
    assert result.returncode == 0, (result.stdout, result.stderr)
    events = [json.loads(l) for l in log.read_text().splitlines()]
    assert events[0]["backend"] == "sim"
    assert [e["stage"] for e in events if e["event"] == "stage"] == ["initial", "step", "return"]
    assert next(e for e in events if e["event"] == "stop")["result"]["stopped"]
    summary = next(e for e in events if e["event"] == "summary")
    assert summary["stages"][-1]["rotation_error_deg"] < .1


@pytest.mark.parametrize("degrees", [0, 6, float("nan"), float("inf")])
def test_probe_rejects_invalid_or_large_steps_before_arm(degrees):
    with pytest.raises(ValueError):
        probe.probe(None, degrees=degrees)
