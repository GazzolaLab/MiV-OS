import numpy as np
import pytest


def test_stimjim_init():
    pytest.importorskip("serial")
    from miv.io.serial import StimjimSerial

    # Test default values
    s = StimjimSerial(port=0)
    assert s.output0_mode == 1
    assert s.output1_mode == 3
    assert s.high_v_1 == 4500
    assert s.low_v_1 == 0
    assert s.high_v_2 == 0
    assert s.low_v_2 == 0

    # Test custom values
    s = StimjimSerial(port=0, output0_mode=2, output1_mode=4, high_v=3000, low_v=1000)
    assert s.output0_mode == 2
    assert s.output1_mode == 4
    assert s.high_v_1 == 3000
    assert s.low_v_1 == 1000
    assert s.high_v_2 == 0
    assert s.low_v_2 == 0


@pytest.fixture
def s():
    pytest.importorskip("serial")
    from miv.io.serial import StimjimSerial

    return StimjimSerial(port=0)


def test_stimjim_send_spiketrain(s):
    # Test default values
    result = s.send_spiketrain(1, [10000, 20000, 30000], 40000, 5)
    expected_result = "S1,1,3,40000,5; 4500,0,10000; 0,0,10000; 4500,0,10000; 0,0,0; 4500,0,10000; 0,0,0; 4500,0,10000; 0,0,0".strip()
    assert result.strip() == expected_result

    # Test reverse parameter
    result = s.send_spiketrain(1, [10000, 20000, 30000], 40000, 5, reverse=True)
    expected_result = "S1,1,3,40000,5; 4500,0,10000; 0,0,10000; 4500,0,10000; 0,0,0; 4500,0,10000; 0,0,0; 4500,0,10000; 0,0,0".strip()
    assert result.strip() == expected_result

    # Test channel parameter
    result = s.send_spiketrain(1, [10000, 20000, 30000], 40000, 5, channel=1)
    expected_result = "S1,1,3,40000,5; 4500,0,10000; 0,0,10000; 4500,0,10000; 0,0,0; 4500,0,10000; 0,0,0; 4500,0,10000; 0,0,0".strip()
    assert result.strip() == expected_result


def test_stimjim_start_str(s):
    # Test default values
    result = s._start_str(1, 1, 3, 4, 5)
    assert result == "S1,1,3,4,5"

    # Test custom values
    result = s._start_str(2, 2, 4, 5, 6)
    assert result == "S2,2,4,5,6"


def test_stimjim_spiketrain_to_str(s):
    # Test default values
    result = s._spiketrain_to_str([1, 2, 3], 4, 1)
    expected_result = (
        [
            "4500,0,1",
            "0,0,1",
            "4500,0,1",
            "0,0,0",
            "4500,0,1",
            "0,0,0",
            "4500,0,1",
            "0,0,0",
        ],
        4,
    )
    assert result == expected_result
