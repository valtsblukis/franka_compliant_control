"""Self-documenting route catalog, rendered without robot reads or external assets."""
from html import escape
import json

from .network import DEFAULT_PORT


DESCRIPTIONS = {
    ("GET", "/"): "Status dashboard. Opening it sends no robot command; recovery requires explicit confirmation.",
    ("GET", "/about"): "This endpoint catalog. Use ?format=json for machine-readable documentation.",
    ("GET", "/urdf"): "Robot-supplied URDF cached at connection startup, in Franky's {urdf: string} JSON format. No robot read on GET. 404 if unavailable.",
    ("GET", "/health"): "Lifecycle/capabilities, selected law, automatic-recovery policy and accepted/applied sequences.",
    ("GET", "/state"): "Full last observation, references, effort, fault and recovery diagnostics. Check sample_age_s.",
    ("GET", "/status"): "Compact dashboard view with joint-limit distances and rejected-operation diagnostics. Cached data only.",
    ("POST", "/state/refresh"): "One fresh inactive robot observation. Does not start control, recover, stop or change settings.",
    ("GET", "/ee_pose"): "Measured TCP pose/twist with freshness and lifecycle metadata.",
    ("GET", "/joint_state"): "Measured joint positions/velocities with freshness and lifecycle metadata.",
    ("GET", "/target_ee_pose"): "Latest accepted Cartesian setpoint, expiry, caller seq and native command_seq. Not a reached flag.",
    ("POST", "/target_ee_pose"): "Submit an absolute base-frame TCP pose; optional twist/acceleration/posture. First valid setpoint starts control. In auto mode selects Cartesian OSC. HTTP 202 means accepted, not reached.",
    ("GET", "/target_joint_state"): "Latest accepted joint setpoint with Franky-compatible fields and native acknowledgement metadata.",
    ("POST", "/target_joint_state"): "Submit seven absolute joint positions and optional velocities. First valid setpoint starts control. In auto mode selects joint impedance. HTTP 200 means accepted, not reached.",
    ("GET", "/controller_config"): "Complete effective controller configuration; also documents tuning and operating limits.",
    ("PUT", "/controller_config"): "Replace configuration while inactive. Omitted fields take schema defaults; read and merge before changing individual settings.",
    ("GET", "/command_timeout"): "Read the finite setpoint watchdog timeout.",
    ("POST", "/command_timeout"): "Set any positive finite command_timeout_s while inactive. Default is 2 seconds, not a maximum. Longer values keep the last target active longer; this does not renew an existing target.",
    ("POST", "/command_timeout/reset"): "Reset streaming command timeout to the 2-second default while inactive. Does not start control or clear a fault.",
    ("GET", "/automatic_recovery"): "Read the live automatic-recovery policy switch.",
    ("POST", "/automatic_recovery"): "Set enabled=true/false for future automatic recovery attempts. Does not execute recovery or cancel an SDK operation already in progress. Enabling is blocked during contact relief or joint-limit backoff; OFF remains allowed.",
    ("POST", "/stop"): "Request stop, reset caller sequence; optional wait_s up to 5 seconds. Stop the producer first: a new setpoint may restart control.",
    ("GET", "/motion_inhibit"): "Read the global process-local software motion inhibit and latest release token. Not a safety-rated e-stop.",
    ("POST", "/motion_inhibit"): "Latch motion inhibition and request stop without waiting for lifecycle work; optional wait_s confirms stopping. Reassertion invalidates prior release tokens.",
    ("POST", "/motion_inhibit/release"): "Release using the latest release_token, only when control/lifecycle work is inactive. Clears targets; never resumes motion or recovers a fault.",
    ("POST", "/recover"): "Explicit inactive recovery. Can move the robot; pause the producer and prepare the workspace first. No target replay.",
    ("POST", "/go_home"): "Start finite nominal-home action using the latest mode. Automatically stops/disarms after measured settling; default 30-second action deadline independent of streaming timeout. New normal setpoints supersede it. HTTP 200 acknowledges the action, not completion. Rejected during contact relief or joint-limit backoff.",
    ("POST", "/relieve_contact"): "Bounded native Cartesian action from writable inactive Disarmed only, with automatic recovery OFF. Default manual mode follows raw SDK wrench without bias or automatic release; requires workspace_confirmed_clear=true for an operator-supervised cleared local retreat/braking area, optional TCP box. Stop explicitly; timeout defaults to 10 s, speeds to 0.005 m/s and 1 deg/s, travel capped at 5 cm / 10 deg. Automatic mode is opt-in and requires verified free-space bias/joints/provenance and an explicit box. HTTP 202 is acceptance, not release or proof of unjamming. No normal targets/home/second relief/joint-limit backoff or enabling recovery while active. Check contact_relief_available/mode/bias_used/status/released in state/health; missing manual bindings/state returns 503.",
    ("POST", "/backoff_joint_limits"): "One finite native joint-limit backoff with strictly inward REFERENCES, not guaranteed inward physical motion or torque (braking can oppose inward velocity); minimal body {confirm_clear_path:true}. No client joints/targets. Native selects from fresh state near/beyond configured effective limits; no selection returns 409 without control. Defaults: trigger_margin=0.05 rad, backoff_distance=0.1 rad (distance minus min(0.02, 0.2*distance) must exceed trigger_margin), speed=0.03 rad/s, torque_scale=1, timeout_s=20 from HTTP receipt. Writable inactive Disarmed or native-eligible LOCAL fault only; native requires fresh stationary error-free Idle and no SDK fault diagnostics. Always bypasses automatic recovery for this action without changing global policy; no OFF prerequisite, recovery call or SDK setters. HTTP 202 means accepted_not_reached, with disarm_on_completion. Check joint_limit_backoff_available/status/reached/selected and native plan in state/health; old bindings/state return 503. No targets/home/contact relief/second backoff or enabling recovery while active. mode_switching_enabled is a configuration capability only: no mode switching during this exclusive action. Selected-joint measured outward encoder/velocity allowance up to 0.01 rad/s REPLACES the ordinary outward directional bound, which may be zero near the specified limit; inward directional and unselected-joint directional bounds remain. Maximum worse-than-start position 0.001 rad / backtracking 0.005 rad are monitored guards, not physical barriers. Global safety configuration and SDK protection remain unchanged. Native refuses an expired pre-control deadline before SDK control entry; a secondary disqualifying fault escalates summary reason while stop_trigger retains the first trigger. Staged implementation, not hardware qualification or current-robot recovery; specified-range/SDK refusal cannot be bypassed.",
    ("POST", "/sim/wrench"): "Optional simulation-only wrench injection; unavailable unless explicitly enabled.",
}


def endpoint_catalog(app, config, *, backend_kind, automatic_recovery, authentication_enabled):
    schema = app.openapi()
    endpoints = []
    for path, operations in schema["paths"].items():
        for method, operation in operations.items():
            if method.upper() not in {"GET", "POST", "PUT", "DELETE", "PATCH"}:
                continue
            body = operation.get("requestBody", {}).get("content", {}).get("application/json", {}).get("schema")
            ref = ((body or {}).get("$ref") or next((item.get("$ref", "") for item in
                   (body or {}).get("anyOf", []) if "$ref" in item), "")).rsplit("/", 1)[-1]
            endpoints.append({"method": method.upper(), "path": path,
                "description": DESCRIPTIONS.get((method.upper(), path), operation.get("summary", "")),
                "request_schema": schema.get("components", {}).get("schemas", {}).get(ref, body),
                "parameters": operation.get("parameters", []),
                "response_status_codes": sorted(operation.get("responses", {}))})
    supported = (["cartesian", "joint"] if config.compliance_mode == "auto" else
                 ["joint"] if config.compliance_mode == "joint_impedance" else
                 ["cartesian"] if config.compliance_mode == "operational_space" else ["cartesian", "joint"])
    return {"service": "franka_compliant_control", "version": app.version,
            "backend": backend_kind, "default_port": DEFAULT_PORT,
            "authentication_enabled": authentication_enabled,
            "compliance_mode": config.compliance_mode, "supported_setpoint_modes": supported,
            "mode_switching_enabled": config.compliance_mode == "auto",
            "automatic_recovery_enabled": automatic_recovery,
            "command_timeout_s": config.command_timeout,
            "frames_and_units": {"position": "absolute base-frame TCP XYZ, metres",
                "orientation": "unit quaternion [x,y,z,w]", "joints": "7 absolute angles, radians",
                "velocities": "7 joint velocities, rad/s",
                "twist": "base-frame [vx,vy,vz,wx,wy,wz], m/s and rad/s",
                "acceleration": "base-frame linear/angular acceleration, m/s^2 and rad/s^2",
                "wrench": "[Fx,Fy,Fz,Mx,My,Mz], N and Nm; frame named in the state field",
                "contact_relief_wrench": "SDK ROBOT-ON-ENVIRONMENT sign, base axes, about configured TCP; external_wrench_tcp_base is nullable, check external_wrench_tcp_valid; relief filtered wrench uses raw readings in manual mode, bias-subtracted readings only in automatic mode"},
            "setpoint_metadata": {"seq": "optional increasing caller sequence, starts at 0; echoed in responses",
                "command_seq": "server-assigned sequence; compare with applied_seq",
                "valid_for_s": "optional lifetime <= command_timeout; otherwise configured timeout is used",
                "deadline_s": "optional absolute server-monotonic expiry instead of valid_for_s; not Unix time"},
            "notes": [
                "No /arm route: first valid setpoint starts a clean inactive controller after native checks, unless motion is inhibited.",
                "Motion inhibit is process-local; explicit token-based release never resumes motion or replays a target.",
                "One latest-setpoint mailbox. Use one coordinated producer; accepted is not reached.",
                "Mode switches in auto configuration use the same native loop and torque slew baseline.",
                "Fresh valid setpoints may attempt guarded automatic recovery if enabled; no old-target replay.",
                "Contact relief never recovers or derives/caches bias. In manual mode, raw estimator bias may cause drift or motion in the wrong direction; slow is not low force. Stop manually; bounds and hard guards remain active. Neither the local retreat bounds nor an optional TCP box are collision planning or a clearance guarantee.",
                "Manual relief reports following_wrench while active, stopped on user stop, and never released=true. State exposes contact_relief_mode and contact_relief_bias_used; acceptance/latest metadata includes relief_options.mode and automatic_release_detection. Automatic release means low estimated residual, not proof of unjamming. No deployed/live support or hardware validation is claimed.",
                "Joint-limit backoff is a general staged action, not a one-off recovery of the current robot. Selected joints alone may start at most 0.05 rad outside configured limits, strictly inside the FR3 specified range; no global limits widen. Native shapes strictly inward references with acceleration min(config,0.05 rad/s^2), jerk min(config,0.5 rad/s^3), friction suppressed, selected active torque caps 2 Nm (J1-4) / 1 Nm (J5-7) times torque_scale bounded by config; other joints keep normal measured-hold feedback budgets. Selected-joint measured outward allowance up to 0.01 rad/s replaces the ordinary outward directional bound (possibly zero); inward and unselected directional bounds, torque and collision guards remain. Inward references do not guarantee inward physical motion or torque; braking may oppose inward velocity. Action guards monitor measured speed over 0.08 rad/s, worsening from start over 0.001 rad, backtracking over 0.005 rad, held-joint drift over 0.01 rad, start mismatch, travel over 0.25 rad, timeout or no progress after 2 s when reference progress is at least 0.02 rad but current actual progress (not peak) is below 0.002 rad. Backtracking high-water marks update through stop observations. Thresholds are monitored guards, not physical barriers; global safety configuration and SDK protection remain unchanged, with no SDK setters. Native blocks automatic recovery after a backoff fault.",
                "Inactive GET polling returns timestamped cached observations. It does not refresh the robot.",
                "URDF is the robot-supplied kinematic model; runtime TCP/payload settings are not synthesized into it.",
                "Normal launchers claim the HTTP port before robot initialization. Default port matches Franky.",
                "No authentication by default: use a trusted robot network. No /about or /urdf GET starts motion."],
            "common_errors": {"409": "invalid lifecycle, motion inhibited/stale release, duplicate/stale sequence, expired command or blocked startup/recovery",
                              "503": "required native capability unavailable; update the inactive backend",
                              "422": "invalid input, unsupported command mode or invalid configuration",
                              "404": "unknown route or unavailable URDF"},
            "links": {"dashboard": "/", "interactive_api": "/docs", "openapi": "/openapi.json",
                      "urdf": "/urdf", "about_json": "/about?format=json"},
            "endpoints": endpoints}


def render_about(catalog):
    rows = []
    for entry in catalog["endpoints"]:
        detail = ""
        for title, value in (("Request body", entry["request_schema"]), ("Query parameters", entry["parameters"])):
            if value:
                detail += f"<details><summary>{title}</summary><pre>{escape(json.dumps(value, indent=2))}</pre></details>"
        rows.append(f"<tr><td><code>{escape(entry['method'])}</code></td><td><code>{escape(entry['path'])}</code></td>"
                    f"<td>{escape(entry['description'])}{detail}</td></tr>")
    notes = "".join(f"<li>{escape(note)}</li>" for note in catalog["notes"])
    summary = {k: v for k, v in catalog.items() if k not in {"endpoints", "notes", "links"}}
    return ("<!doctype html><html lang='en'><head><meta charset='utf-8'>"
            "<meta name='viewport' content='width=device-width,initial-scale=1'><title>Controller API</title>"
            "<style>body{max-width:1100px;margin:24px auto;padding:0 16px;font-family:system-ui;background:#10151e;color:#e5ecf5}"
            "a{color:#8ac2ff}table{width:100%;border-collapse:collapse}td,th{text-align:left;vertical-align:top;padding:12px;border-bottom:1px solid #445}"
            "pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#182130;padding:12px}li{margin:8px 0}details{margin-top:8px}code{white-space:nowrap}</style></head><body>"
            "<h1>Franka controller HTTP API</h1><p><a href='/'>Dashboard</a> · <a href='/docs'>Interactive API</a> · "
            "<a href='/about?format=json'>JSON catalog</a> · <a href='/urdf'>Robot URDF</a></p>"
            f"<ul>{notes}</ul><details><summary>Configuration, units and command semantics</summary><pre>{escape(json.dumps(summary, indent=2))}</pre></details>"
            "<h2>Endpoints</h2><table><thead><tr><th>Method</th><th>Path</th><th>Meaning / input</th></tr></thead>"
            f"<tbody>{''.join(rows)}</tbody></table></body></html>")
