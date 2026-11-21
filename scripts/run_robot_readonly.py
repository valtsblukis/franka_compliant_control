"""Start the local commissioning service with native arming/recovery disabled."""
from pathlib import Path

from franka_compliant_control.network import run_http_server
from franka_compliant_control.models import ControllerConfig, HardwareSettings
from franka_compliant_control.service import create_app

root = Path(__file__).resolve().parents[1]
hardware = HardwareSettings.model_validate_json((root / "local-hardware.readonly.json").read_text())
if not hardware.read_only:
    raise RuntimeError("This launcher requires read_only=true")
config = ControllerConfig.model_validate_json((root / "local-controller.json").read_text())
pid_path = root / "logs/readonly-service.pid"
run_http_server(create_app(config=config, backend_kind="robot", hardware=hardware, automatic_recovery=False),
                host="127.0.0.1", pid_path=pid_path)
