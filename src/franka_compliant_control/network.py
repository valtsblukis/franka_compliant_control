"""Shared Franky-compatible endpoint and bind-before-robot startup."""
from contextlib import contextmanager
import logging
import os
import socket

DEFAULT_PORT = 54321
DEFAULT_URL = f"http://127.0.0.1:{DEFAULT_PORT}"


@contextmanager
def reserve_listener(host, port):
    """Hold the actual listening socket; no check-then-close startup race.

    SO_REUSEPORT is deliberately not enabled. A conflicting listener fails
    before Uvicorn enters the app lifespan (and therefore before robot I/O).
    """
    family = socket.AF_INET6 if ":" in host else socket.AF_INET
    try:
        listener = socket.create_server((host, port), family=family, backlog=2048, reuse_port=False)
    except OSError as error:
        raise RuntimeError(f"Cannot claim {host}:{port}: {error}. No robot connection was attempted. "
                           "Check whether franky_service or another controller already owns this endpoint.") from error
    try:
        yield listener
    finally:
        listener.close()


def run_http_server(app, *, host="127.0.0.1", port=DEFAULT_PORT, pid_path=None):
    import uvicorn

    with reserve_listener(host, port) as listener:
        if pid_path is not None:
            pid_path.parent.mkdir(parents=True, exist_ok=True)
            pid_path.write_text(str(os.getpid())+"\n")
        try:
            config = uvicorn.Config(app, host=host, port=port, workers=1)
            logging.getLogger("uvicorn.error").info(
                "Claimed HTTP endpoint %s before controller initialization", listener.getsockname())
            # Pass the same held socket, rather than asking Uvicorn to rebind
            # after lifespan startup. No reload or multiworker supervisor.
            try:
                uvicorn.Server(config).run(sockets=[listener])
            except KeyboardInterrupt:
                pass  # Match uvicorn.run's foreground Ctrl+C behavior.
        finally:
            if pid_path is not None and pid_path.exists() and pid_path.read_text().strip() == str(os.getpid()):
                pid_path.unlink()
