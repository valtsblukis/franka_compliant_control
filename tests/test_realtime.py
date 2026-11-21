from types import SimpleNamespace

import pytest

from franka_compliant_control import _native
from franka_compliant_control.service import _check_realtime_kernel


@pytest.mark.parametrize("indicator,version,sdk,accepted", [
    ("1", "PREEMPT_RT", True, True),
    ("0", "PREEMPT_RT", True, False),
    ("garbage", "PREEMPT_RT", True, False),
    (None, "#1 SMP PREEMPT_RT Tue Sep 9", True, True),
    (None, "#1 SMP PREEMPT_DYNAMIC", True, False),
    (None, "#1 SMP PREEMPT", True, False),
    (None, "#1 SMP NOT_PREEMPT_RT", True, False),
    (None, "#1 SMP PREEMPT_RT", False, False),
])
def test_kernel_and_linked_sdk_must_recognize_rt(monkeypatch, indicator, version, sdk, accepted):
    def read(_self):
        if indicator is None:
            raise FileNotFoundError
        return indicator
    monkeypatch.setattr("pathlib.Path.read_text", read)
    monkeypatch.setattr("os.uname", lambda: SimpleNamespace(version=version, release="6.18-rt"))
    monkeypatch.setattr(_native, "has_realtime_kernel", lambda: sdk)
    if accepted:
        _check_realtime_kernel()
    else:
        with pytest.raises(RuntimeError):
            _check_realtime_kernel()
