import pytest
from franka_compliant_control.models import HardwareSettings


def settings(**changes):
    return HardwareSettings(robot_ip="unused", robot_model="fr3",
        flange_to_tcp=[1,0,0,0,0,1,0,0,0,0,1,0,0,0,.174,1],
        load_mass=0, load_com=[0]*3, load_inertia=[0]*9, **changes)


TOOL = {"mass": .9, "com": [0,0,.057],
        "inertia": [.002768,0,0,0,.003149,0,0,0,.000564]}


@pytest.mark.parametrize("basis", ["manufacturer_nominal", "estimated"])
def test_nominal_requires_provenance_and_expected_tool(basis):
    with pytest.raises(ValueError, match="source and expected_end_effector"):
        settings(configuration_basis=basis)
    with pytest.raises(ValueError, match="source and expected_end_effector"):
        settings(configuration_basis=basis, configuration_source=" ", expected_end_effector=TOOL)
    with pytest.raises(ValueError, match="measured calibration"):
        settings(configuration_basis=basis, configuration_source="Test tool provenance",
                 expected_end_effector=TOOL, calibration_confirmed=True)
    config = settings(configuration_basis=basis, configuration_source="Test tool provenance",
                      expected_end_effector=TOOL)
    assert not config.calibration_confirmed
    native = config.native()
    assert native.check_end_effector
    native.validate_end_effector(.899999976, [0,0,.05700000003], TOOL["inertia"])
    with pytest.raises(RuntimeError, match="Desk end-effector"):
        native.validate_end_effector(.73, [-.01,0,.03], [.001,0,0,0,.0025,0,0,0,.0017])


@pytest.mark.parametrize("mass,com,inertia", [
    (float("nan"), TOOL["com"], TOOL["inertia"]),
    (.9, [0,0,.06], TOOL["inertia"]),
    (.9, TOOL["com"], [.00418,0,0,0,.00508,0,0,0,.00125]),
])
def test_native_tool_guard_rejects_mismatch(mass, com, inertia):
    config = settings(configuration_basis="manufacturer_nominal", configuration_source="Franka example",
                      expected_end_effector=TOOL).native()
    with pytest.raises(RuntimeError, match="Desk end-effector"):
        config.validate_end_effector(mass, com, inertia)
