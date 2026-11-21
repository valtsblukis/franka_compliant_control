#!/usr/bin/env bash
# Unified HTTP controller + dashboard. Setpoint type selects joint impedance or OSC.
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
exec "$ROOT/.venv/bin/python" "$ROOT/scripts/run_robot.py" \
  --config "$ROOT/configs/controller.fr3.auto.json" \
  --host 0.0.0.0 "$@"
