"""Pure Python relief schema/client tests: no native module, server or network."""
import math

import pytest
import requests
from pydantic import ValidationError

from franka_compliant_control.client import CompliantClient
from franka_compliant_control.models import ContactReliefRequest


OPTIONAL_VECTORS = {
    "bias_wrench_tcp_base": None, "bias_joint_positions": None,
    "workspace_lower": None, "workspace_upper": None,
}


def test_minimal_manual_request_defaults_do_not_invent_bias_or_box():
    request = ContactReliefRequest(workspace_confirmed_clear=True)
    assert request.mode == "manual" and request.timeout_s == 10
    assert request.translation_speed == .005 and request.rotation_speed == math.radians(1)
    assert request.max_translation == .05 and request.max_rotation == math.radians(10)
    assert request.bias_source is None and not request.bias_confirmed_free
    assert all(getattr(request, name) is None for name in OPTIONAL_VECTORS)
    assert request.model_dump() == ContactReliefRequest(
        workspace_confirmed_clear=True, bias_source=None, **OPTIONAL_VECTORS).model_dump()
    assert ContactReliefRequest.model_json_schema()["required"] == ["workspace_confirmed_clear"]


@pytest.mark.parametrize("ack", [None, False, 0, 1, "true"])
def test_manual_requires_strict_clearance_acknowledgment(ack):
    with pytest.raises(ValidationError):
        ContactReliefRequest(workspace_confirmed_clear=ack)
    with pytest.raises(ValidationError):
        ContactReliefRequest()


@pytest.mark.parametrize("changes", [
    {"bias_wrench_tcp_base": [0.] * 6}, {"bias_joint_positions": [0.] * 7},
    {"bias_source": "verified record"}, {"bias_confirmed_free": True},
    {"workspace_lower": [0.] * 3}, {"workspace_upper": [1.] * 3},
    {"workspace_lower": [0.] * 3, "workspace_upper": [0.] * 3},
    {"workspace_lower": [0., 2., 0.], "workspace_upper": [1.] * 3},
    {"workspace_lower": [0.] * 3, "workspace_upper": [float("nan")] * 3},
    {"workspace_lower": [0.] * 3, "workspace_upper": [1.] * 2},
    {"mode": "other"}, {"mode": None}, {"manual_stop": True},
    {"automatic_release_detection": False},
])
def test_manual_rejects_bias_and_invalid_optional_workspace(changes):
    with pytest.raises(ValidationError):
        ContactReliefRequest(workspace_confirmed_clear=True, **changes)


def test_manual_accepts_explicit_ordered_workspace():
    request = ContactReliefRequest(workspace_confirmed_clear=True,
                                  workspace_lower=[.4, -.1, .3], workspace_upper=[.5, .1, .4])
    assert request.mode == "manual" and request.bias_wrench_tcp_base is None
    assert request.workspace_lower == [.4, -.1, .3] and request.workspace_upper == [.5, .1, .4]


@pytest.mark.parametrize("missing", ["bias_wrench_tcp_base", "bias_joint_positions", "bias_source",
                                     "bias_confirmed_free", "workspace_lower", "workspace_upper", "box"])
def test_automatic_still_requires_verified_bias_and_explicit_box(missing):
    payload = dict(mode="automatic", workspace_confirmed_clear=True,
                   bias_wrench_tcp_base=[0.] * 6, bias_joint_positions=[0.] * 7,
                   bias_source="synthetic test-only verified-free record", bias_confirmed_free=True,
                   workspace_lower=[0.] * 3, workspace_upper=[1.] * 3)
    assert ContactReliefRequest(**payload).mode == "automatic"
    for name in (("workspace_lower", "workspace_upper") if missing == "box" else (missing,)):
        del payload[name]
    with pytest.raises(ValidationError):
        ContactReliefRequest(**payload)


@pytest.mark.parametrize("options", [{}, {**OPTIONAL_VECTORS, "bias_source": None},
                                    {"workspace_lower": (.4, -.1, .3), "workspace_upper": (.5, .1, .4)}])
def test_manual_client_sends_once_with_defaults_and_optional_none_vectors(monkeypatch, options):
    calls = []

    def request(self, method, path, **kwargs):
        calls.append((method, path, kwargs))
        return {"accepted": True}

    monkeypatch.setattr(CompliantClient, "_request", request)
    with CompliantClient() as client:
        assert client.relieve_contact(workspace_confirmed_clear=True, **options)["accepted"]
        assert client._sent_setpoints
        assert [(method, path) for method, path, _ in calls] == [("POST", "/relieve_contact")]
        payload = calls[0][2]["json"]
        assert payload == ContactReliefRequest(workspace_confirmed_clear=True, **options).model_dump(mode="json")
        assert payload["mode"] == "manual" and payload["timeout_s"] == 10
        assert payload["bias_source"] is None and payload["bias_wrench_tcp_base"] is None
        assert payload["bias_joint_positions"] is None and not payload["bias_confirmed_free"]
    assert [path for _, path, _ in calls] == ["/relieve_contact", "/stop"]


@pytest.mark.parametrize("options", [{}, {"workspace_confirmed_clear": False},
                                    {"workspace_confirmed_clear": True, "bias_confirmed_free": True},
                                    {"workspace_confirmed_clear": True, "workspace_lower": None,
                                     "workspace_upper": [1.] * 3}])
def test_invalid_manual_client_request_never_sends(monkeypatch, options):
    def unexpected(*args, **kwargs):
        pytest.fail("Invalid request must not send any HTTP call")

    monkeypatch.setattr(CompliantClient, "_request", unexpected)
    with CompliantClient() as client:
        with pytest.raises(ValueError):
            client.relieve_contact(**options)
        assert not client._sent_setpoints


def test_manual_client_never_retries_or_recovers_lost_acceptance(monkeypatch):
    calls = []

    def request(self, method, path, **kwargs):
        calls.append(path)
        if path == "/relieve_contact":
            raise requests.Timeout("acceptance response lost")
        return {}

    monkeypatch.setattr(CompliantClient, "_request", request)
    with CompliantClient() as client:
        with pytest.raises(requests.Timeout):
            client.relieve_contact(workspace_confirmed_clear=True)
        assert calls == ["/relieve_contact"] and client._sent_setpoints
    assert calls == ["/relieve_contact", "/stop"]
