#!/usr/bin/env bash
# Convenience alias for the unified service; joint or Cartesian setpoints select the law.
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec "$ROOT/start.sh" "$@"
