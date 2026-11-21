import importlib.util
from pathlib import Path
import pytest


def test_task_reference_recovers_in_seconds_instead_of_creeping():
    spec = importlib.util.spec_from_file_location("benchmark_reference", "scripts/benchmark_reference.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    old = module.benchmark(Path("tests/fixtures/controller/controller.fr3.perturbation-20260909-v2.json"))
    new = module.benchmark(Path("configs/controller.fr3.perturbation.json"))
    assert old["translation_10cm_reference_s"] > 4
    assert new["translation_10cm_reference_s"] < 1.3
    assert old["rotation_20deg_reference_s"] > 3
    assert new["rotation_20deg_reference_s"] < 1.7
    assert old["translation_reached_speed_limit"] and new["translation_reached_speed_limit"]
    with pytest.raises(ValueError, match="only applies to contact_yielding"):
        module.benchmark(Path("configs/controller.fr3.impedance.json"))
