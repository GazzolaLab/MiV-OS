import numpy as np
import pytest

from miv.mea.channel_mapping import MEA128


@pytest.fixture
def mea():
    return MEA128()


def test_channel_mapping(mea):
    assert mea.channel_mapping(1) == 7
    assert mea.channel_mapping(7, reverse=True) == 1


@pytest.mark.parametrize(
    "oe_channel, intan_str",
    [
        (33, "B-1"),
        (65, "C-1"),
        (31, "A-31"),
        (64, "C-0"),
    ],
)
def test_intan_channel_mapping_loop(oe_channel, intan_str, mea):
    assert mea.intan_channel_int_to_str(oe_channel) == intan_str
    assert mea.intan_channel_str_to_int(intan_str) == oe_channel
    assert (
        mea.intan_channel_str_to_int(mea.intan_channel_int_to_str(oe_channel))
        == oe_channel
    )
    assert (
        mea.intan_channel_int_to_str(mea.intan_channel_str_to_int(intan_str))
        == intan_str
    )


def test_plot_network(mea):
    # TODO
    # for now, just test that the code runs without errors
    channels = [1, 2, 3, 4]
    mea.plot_network(channels)
