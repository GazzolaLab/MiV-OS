import pytest

from miv.mea.core import _MEA, MEA


def test_singleton_mea_module():
    mea1 = _MEA()
    mea2 = _MEA()
    assert mea1 is mea2
    assert MEA is mea2
    assert mea1 is MEA


def test_mea_call_run():
    MEA()


def test_mea_instance():
    assert MEA._instance is not None


def test_mea_register_and_get_electrode_path(tmp_path):
    # Make test.yaml file in tmp_path
    tmp_file = tmp_path / "test.yaml"
    tmp_file.touch()

    # Register tmp_path
    MEA.register_electrode_path(tmp_path)
    assert tmp_path in MEA._reg_electrode_paths

    # Get electrode paths and check if tmp_file is in
    electrode_paths = MEA.get_electrode_paths()
    assert str(tmp_file) in electrode_paths


def test_mea_build_from_dictionary(tmp_path):
    name = "TestABC"
    test_mea_info = {
        "electrode_name": name,
        "description": "Test MEA",
        "sortlist": None,
        "pitch": [15.0, 32.0],
        "dim": [32, 2],
        "size": 6.0,
        "plane": "yz",
        "shape": "square",
        "type": "mea",
    }

    mea = MEA.return_mea_from_dict(info=test_mea_info)
    assert mea.info["electrode_name"] == name

    # Save in yaml
    import yaml

    with open(tmp_path / "test_mea.yaml", "w") as f:
        yaml.dump(test_mea_info, f, default_flow_style=False)

    # register
    MEA.register_electrode_path(tmp_path)
    assert tmp_path in MEA._reg_electrode_paths

    # check mea names
    assert name in MEA.return_mea_list()

    # import mea
    imported_info = MEA.return_mea_info(name)
    assert imported_info["electrode_name"] == name

    # check mea
    mea = MEA.return_mea(name)
    assert mea.info["electrode_name"] == name


@pytest.mark.parametrize("mea_name", ["64_intanRHD"])
def test_miv_mea_exist_for_default(mea_name):
    mea = MEA.return_mea(mea_name)
    assert mea.info["electrode_name"] == "64_intanRHD"
