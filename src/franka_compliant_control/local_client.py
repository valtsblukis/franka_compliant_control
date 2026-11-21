"""Own a native controller in the caller's process, without an HTTP server."""
from . import _native
from .models import CartesianTarget, ControllerConfig, HardwareSettings
from .service import ControllerService, _check_realtime_kernel


class LocalControllerClient:
    """The hold client interface with one explicitly owned native connection.

    Construction connects but does not arm. Closing joins the native worker and
    releases the robot connection. No subprocess, server, recovery or retry.
    """

    def __init__(self, config: ControllerConfig, *, backend_kind="robot",
                 hardware: HardwareSettings | None = None):
        if backend_kind not in ("robot", "sim"):
            raise ValueError("backend_kind must be robot or sim")
        if backend_kind == "robot":
            if hardware is None or hardware.read_only:
                raise ValueError("Standalone hold requires writable hardware settings")
            if hardware.robot_model != config.robot_model:
                raise ValueError("Hardware and controller robot_model must match")
            _check_realtime_kernel()
        backend = _native.Backend(config.native(), backend_kind,
                                  hardware.native() if hardware else _native.HardwareConfig())
        self.controller = ControllerService(backend, config)
        self.final_state = None
        self.final_state_error = None
        self.initialized = False
        self.sequence = 0
        self.valid_for_s = min(.15, .8 * config.command_timeout)

    def state(self):
        return self.controller.state()

    def get_config(self):
        return self.controller.config.model_dump()

    def arm(self):
        result = self.controller._initialize_local_hold()
        self.initialized = True
        self.sequence = 0
        return result

    def send_pose(self, position, quaternion_xyzw, *, twist=None, acceleration=None, posture=None):
        if not self.initialized:
            raise RuntimeError("Initialize the standalone hold before sending commands")
        self.sequence += 1
        return self.controller.submit(CartesianTarget(
            seq=self.sequence,
            deadline_s=_native.monotonic_time() + self.valid_for_s,
            position=list(position), quaternion_xyzw=list(quaternion_xyzw),
            twist=list(twist) if twist is not None else [0.0]*6,
            acceleration=list(acceleration) if acceleration is not None else [0.0]*6,
            posture=list(posture) if posture is not None else None), start_if_inactive=False)

    def stop(self, wait_s=1.0):
        self.initialized = False
        return self.controller.stop(wait_s)

    def get_motion_inhibit(self):
        return self.controller.motion_inhibit_status()

    def inhibit_motion(self, wait_s=0.0):
        self.initialized = False
        return self.controller.inhibit_motion(wait_s)

    def release_motion_inhibit(self, release_token):
        return self.controller.release_motion_inhibit(release_token)

    def close(self):
        controller, self.controller = self.controller, None
        self.initialized = False
        if controller is not None:
            try:
                controller.backend.close()
            finally:
                try:
                    # Preserve late post-abort diagnostics after joining the
                    # worker, even if an earlier stop(wait_s) timed out.
                    if hasattr(controller.backend, "state"):
                        self.final_state = controller.state()
                except Exception as error:
                    self.final_state_error = str(error)  # Do not mask a close failure.
                finally:
                    # An exception traceback must not retain the connection.
                    controller.backend = None

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
