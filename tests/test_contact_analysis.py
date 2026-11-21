import json
import math

import pytest

from franka_compliant_control.contact_analysis import analyze_log


def write_log(path, state):
    original = {"compliance_mode": "continuous_impedance", "contact_force_stop": 25,
                "contact_moment_stop": 10, "external_torque_limits": [8]*7}
    events = [{"event": "launch", "controller_config": original},
              {"event": "sample", "state": state},
              {"event": "stop", "result": {"state": {**state, "stop_sample": None,
                                                       "stop_trigger": "backend_error"}}}]
    path.write_text("\n".join(json.dumps(e) for e in events))
    return {**original, "contact_force_stop": 40, "contact_moment_stop": 20,
            "external_torque_limits": [16]*4+[8]*3}


def test_comparison_uses_tcp_axes_and_retains_missing_trigger_limitation(tmp_path):
    state = {"external_wrench": [0,0,0,9,9,0], "external_joint_torques": [0]*7,
             "quaternion_xyzw": [0,0,math.sin(math.pi/8),math.cos(math.pi/8)]}
    path = tmp_path/"contact.jsonl"
    candidate = write_log(path, state)
    result = analyze_log(path, candidate)
    assert result["peak_tcp_moment_components_Nm"] == pytest.approx([9*math.sqrt(2),0,0])
    assert result["recorded_config_peak_threshold_fraction"] == pytest.approx(9*math.sqrt(2)/10)
    assert result["candidate_peak_threshold_fraction"] == pytest.approx(9*math.sqrt(2)/20)
    assert result["candidate_threshold_crossing_observations"] == 0
    assert not result["native_trigger_sample_present"]
    assert result["stop_trigger"] == "backend_error" and "omitted" in result["limitation"]


def test_comparison_does_not_drop_an_unsampled_native_trigger(tmp_path):
    state = {"external_wrench": [0]*6, "external_joint_torques": [0]*7,
             "quaternion_xyzw": [0,0,0,1]}
    path = tmp_path/"contact.jsonl"
    candidate = write_log(path, state)
    events = [json.loads(line) for line in path.read_text().splitlines()]
    events[-1]["result"]["state"]["stop_sample"] = {**state, "external_joint_torques": [0]*6+[8.1]}
    path.write_text("\n".join(json.dumps(e) for e in events))
    result = analyze_log(path, candidate)
    assert result["candidate_threshold_crossing_observations"] == 1
    assert result["candidate_peak_threshold_fraction"] == pytest.approx(8.1/8)
    assert result["native_trigger_sample_present"]


def test_inherited_robot_limits_are_unknown_not_taken_from_unused_config(tmp_path):
    state = {"external_wrench": [20,20,0,0,0,0], "external_joint_torques": [0]*7,
             "quaternion_xyzw": [0,0,0,1]}
    path = tmp_path/"inherited.jsonl"
    candidate = write_log(path, state)
    events = [json.loads(line) for line in path.read_text().splitlines()]
    events[0]["controller_config"].update(compliance_mode="operational_space", collision_policy="robot")
    path.write_text("\n".join(json.dumps(e) for e in events))
    result = analyze_log(path, {**candidate, "compliance_mode": "operational_space"})
    assert result["recorded_config_peak_threshold_fraction"] is None
    assert result["candidate_peak_threshold_fraction"] == .5  # Per-axis, not resultant.


def test_sdk_records_and_acknowledged_limits_are_used_without_double_counting(tmp_path):
    state = {"external_wrench": [0]*6, "external_joint_torques": [0]*7, "quaternion_xyzw": [0,0,0,1]}
    path = tmp_path/"sdk.jsonl"
    candidate = write_log(path, state)
    events = [json.loads(line) for line in path.read_text().splitlines()]
    events[0]["controller_config"].update(compliance_mode="operational_space", collision_policy="robot_configured")
    stop = events[-1]["result"]["state"]
    stop["robot_collision_settings"] = {"status": "applied", "thresholds": {
        "upper_force_thresholds_nominal": [40]*3+[20]*3,
        "upper_force_thresholds_acceleration": [40]*3+[20]*3,
        "upper_torque_thresholds_nominal": [16]*4+[8]*3,
        "upper_torque_thresholds_acceleration": [16]*4+[8]*3}}
    # SDK K-frame measurement is preserved directly, not guessed from the pose.
    sdk = {**state, "external_wrench_stiffness": [41,0,0,0,0,0]}
    stop["fault_diagnostics"] = {"caught_at_monotonic_s": 1,
        "sdk_records": [{"state": sdk}], "post_fault_observation": {
            "fresh": True, "valid": True, "sampled_at_monotonic_s": 2, "state": state}}
    events.append({"event": "controller_closed", "state": stop})
    path.write_text("\n".join(json.dumps(e) for e in events))
    result = analyze_log(path, candidate)
    assert result["sdk_exception_record_observations"] == 1
    assert result["post_fault_observation_present"]
    assert result["candidate_threshold_crossing_observations"] == 1
    assert result["recorded_config_peak_threshold_fraction"] == 41/40
    assert result["recorded_threshold_source"] == "sdk_acknowledged_request"
    assert not result["native_trigger_sample_present"]


@pytest.mark.parametrize("change", [{"quaternion_xyzw": [0,0,0,2]},
                                    {"external_wrench": [float("nan")]*6},
                                    {"sample_valid": False}])
def test_invalid_data_cannot_produce_a_successful_comparison(tmp_path, change):
    state = {"external_wrench": [0]*6, "external_joint_torques": [0]*7,
             "quaternion_xyzw": [0,0,0,1], **change}
    path = tmp_path/"invalid.jsonl"
    candidate = write_log(path, state)
    with pytest.raises(ValueError):
        analyze_log(path, candidate)
