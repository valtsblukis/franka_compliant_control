"""Run the real HTTP client against an independently spawned mock service."""
import os
import secrets
import socket
import subprocess
import sys
import time

import requests

from franka_compliant_control.client import CompliantClient


def test_streaming_over_http_and_lost_command_stop():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    token = secrets.token_urlsafe(32)
    env = dict(os.environ, FCC_API_TOKEN=token)
    process = subprocess.Popen(
        [sys.executable, "-m", "franka_compliant_control.service", "--backend", "sim", "--port", str(port)],
        env=env, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
    )
    # The old environment variable must not implicitly enable authentication.
    client = CompliantClient(f"http://127.0.0.1:{port}", timeout_s=1)
    try:
        for _ in range(100):
            assert process.poll() is None, "mock service exited during startup"
            try:
                initial = client.state()
                break
            except requests.ConnectionError:
                time.sleep(.02)
        else:
            raise AssertionError("mock service did not start")
        assert initial["backend"] == "sim"
        position = list(initial["position"])
        position[0] += .005
        samples = []
        for _ in range(100):
            client.send_pose(position)
            samples.append(client.state())
            time.sleep(.02)
        assert all(s["phase"] == "active" for s in samples[2:])
        assert samples[-1]["position"][0] > initial["position"][0] + .002
        assert abs(samples[-1]["position"][0] - position[0]) < .003
        time.sleep(2.5)  # stop producing targets; no /stop call should be needed
        state = client.state()
        assert not state["running"] and state["reason"] == "command_timeout"
    finally:
        client._sent_setpoints = False
        client.close()
        process.terminate()
        try:
            process.wait(timeout=4)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=2)
        process.stderr.close()
