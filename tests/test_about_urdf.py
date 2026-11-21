"""Read-only API documentation and cached-model tests; recording backend only."""
from pathlib import Path

from fastapi.testclient import TestClient

from franka_compliant_control.about import render_about
from franka_compliant_control.client import CompliantClient
from franka_compliant_control.models import ControllerConfig
from franka_compliant_control.service import create_app
from test_policy_http import service, TOKEN


def test_urdf_returns_exact_cached_robot_model_without_lifecycle_operations(service):
    http, backend, _ = service
    xml = '<?xml version="1.0"?><robot name="fixture"><link name="base"/></robot>'
    backend.urdf_cache = xml
    for phase, running in (("disarmed",False),("active",True),("fault",False)):
        backend.phase, backend.running = phase, running
        response = http.get("/urdf")
        assert response.status_code == 200
        assert response.json() == {"urdf": xml}
    assert backend.arm_calls == backend.start_calls == backend.submit_calls == backend.recover_calls == 0


def test_missing_urdf_is_explicit_not_a_fabricated_model(service):
    http, backend, _ = service
    response = http.get("/urdf")
    assert response.status_code == 404 and "no saved-model fallback" in response.json()["detail"]
    assert backend.arm_calls == backend.submit_calls == 0


def test_about_matches_registered_endpoints_and_current_capabilities(service):
    http, backend, _ = service
    response = http.get("/about")
    assert response.status_code == 200 and "text/html" in response.headers["content-type"]
    assert "/target_joint_state" in response.text and "/target_ee_pose" in response.text
    catalog = http.get("/about?format=json").json()
    actual = {(method.upper(), path) for path, operations in http.get("/openapi.json").json()["paths"].items()
              for method in operations}
    listed = {(entry["method"],entry["path"]) for entry in catalog["endpoints"]}
    assert actual == listed and ("POST","/arm") not in listed
    assert catalog["default_port"] == 54321 and not catalog["authentication_enabled"]
    assert catalog["supported_setpoint_modes"] == ["cartesian"]
    assert catalog["command_timeout_s"] == 2.0
    joint = next(e for e in catalog["endpoints"] if e["method"] == "POST" and e["path"] == "/target_joint_state")
    assert joint["request_schema"]["required"] == ["positions"]
    assert "200" in joint["response_status_codes"]
    config = ControllerConfig.model_validate_json(Path("configs/controller.fr3.auto.json").read_text())
    http.put("/controller_config", json=config.model_dump())
    assert http.get("/about?format=json").json()["supported_setpoint_modes"] == ["cartesian","joint"]
    assert backend.arm_calls == backend.submit_calls == backend.recover_calls == 0


def test_about_html_escapes_description_text(service):
    http, _, _ = service
    catalog = http.get("/about?format=json").json()
    catalog["notes"] = ['<script>alert("bad")</script>']
    catalog["endpoints"][0]["description"] = '<img src=x onerror="bad()">'
    rendered = render_about(catalog)
    assert "<script>" not in rendered and "<img" not in rendered
    assert "&lt;script&gt;" in rendered and "&lt;img" in rendered


def test_urdf_and_about_respect_explicit_custom_auth(service):
    with TestClient(create_app(api_token=TOKEN)) as http:
        for path in ("/about", "/urdf"):
            assert http.get(path).status_code == 401
        http.headers["Authorization"] = f"Bearer {TOKEN}"
        assert http.get("/about?format=json").json()["authentication_enabled"]
        assert http.get("/urdf").status_code == 404


def test_client_documentation_and_urdf_helpers_do_not_send_commands(service):
    http, backend, _ = service
    backend.urdf_cache = '<robot name="test"/>'
    client = CompliantClient("http://testserver")
    client.http.close();client.http = http
    assert client.get_urdf() == {"urdf": backend.urdf_cache}
    assert client.about()["service"] == "franka_compliant_control"
    assert not client._sent_setpoints and backend.arm_calls == backend.submit_calls == 0
