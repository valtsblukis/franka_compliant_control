"""Socket-exclusion/startup-order tests; no robot, controller or simulated plant."""
import importlib.util
import os
from pathlib import Path
import socket
from types import SimpleNamespace

import pytest
import uvicorn

from franka_compliant_control.client import CompliantClient
from franka_compliant_control.network import DEFAULT_PORT, DEFAULT_URL, reserve_listener, run_http_server


def test_defaults_match_franky_port():
    assert DEFAULT_PORT == 54321 and DEFAULT_URL == "http://127.0.0.1:54321"
    client = CompliantClient()
    assert client.base_url == DEFAULT_URL
    client.close()


@pytest.mark.parametrize("first,second", [("0.0.0.0","0.0.0.0"), ("0.0.0.0","127.0.0.1"), ("127.0.0.1","0.0.0.0")])
def test_listener_excludes_another_default_service(first, second):
    with reserve_listener(first, 0) as listener:
        port = listener.getsockname()[1]
        if hasattr(socket, "SO_REUSEPORT"):
            assert listener.getsockopt(socket.SOL_SOCKET, socket.SO_REUSEPORT) == 0
        with pytest.raises(RuntimeError, match="No robot connection was attempted"):
            with reserve_listener(second, port):
                pytest.fail("conflicting listener was admitted")
    with reserve_listener(first, port):
        pass  # Socket ownership was released cleanly.


def test_conflict_prevents_server_lifespan_and_preserves_existing_pid(tmp_path, monkeypatch):
    pid = tmp_path/"robot-service.pid";pid.write_text("existing-owner\n")
    def forbidden(*_, **__):
        pytest.fail("Uvicorn/robot lifespan must not start before acquiring the socket")
    monkeypatch.setattr(uvicorn, "Config", forbidden)
    with reserve_listener("0.0.0.0", 0) as existing:
        with pytest.raises(RuntimeError, match="already owns"):
            run_http_server(object(), host="0.0.0.0", port=existing.getsockname()[1], pid_path=pid)
    assert pid.read_text() == "existing-owner\n"


@pytest.mark.parametrize("fail", [False, True])
def test_uvicorn_receives_the_held_socket_and_cleanup_runs(tmp_path, monkeypatch, fail):
    pid = tmp_path/"subdir/robot-service.pid"
    captured = {}
    app = object()
    class Server:
        def __init__(self, config):
            assert config.app is app and config.workers == 1
        def run(self, sockets):
            assert len(sockets) == 1
            captured["socket"] = sockets[0]
            captured["port"] = sockets[0].getsockname()[1]
            assert pid.read_text().strip() == str(os.getpid())
            with pytest.raises(RuntimeError):
                with reserve_listener("127.0.0.1", captured["port"]):
                    pass
            if fail:
                raise RuntimeError("mock startup failure")
    monkeypatch.setattr(uvicorn, "Config", lambda app, **kw: SimpleNamespace(app=app, **kw))
    monkeypatch.setattr(uvicorn, "Server", Server)
    if fail:
        with pytest.raises(RuntimeError, match="mock startup failure"):
            run_http_server(app, port=0, pid_path=pid)
    else:
        run_http_server(app, port=0, pid_path=pid)
    assert captured["socket"].fileno() == -1 and not pid.exists()
    with reserve_listener("127.0.0.1", captured["port"]):
        pass


def test_hardware_launcher_fails_before_native_connection_on_port_conflict(tmp_path, monkeypatch):
    spec = importlib.util.spec_from_file_location("bound_launcher", "scripts/run_robot.py")
    module = importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    monkeypatch.setattr(module, "ROOT", tmp_path)
    (tmp_path/"local-hardware.json").write_text(Path("configs/hardware.wrist-load.estimated.json").read_text())
    (tmp_path/"local-controller.json").write_text(Path("configs/controller.fr3.auto.json").read_text())
    def forbidden(*_, **__):
        pytest.fail("No server/robot startup should happen with a conflicting listener")
    monkeypatch.setattr(uvicorn, "Server", forbidden)
    with reserve_listener("0.0.0.0", 0) as existing:
        with pytest.raises(SystemExit) as error:
            module.main(["--host","0.0.0.0","--port",str(existing.getsockname()[1])])
        assert error.value.code == 1
    assert not (tmp_path/"logs/robot-service.pid").exists()
