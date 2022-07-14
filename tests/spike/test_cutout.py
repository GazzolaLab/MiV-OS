import numpy as np

from miv.signal.spike.cutout import ChannelSpikeCutout, SpikeCutout


class MockSpikeCutout(SpikeCutout):
    def __init__(
        self, spike_type: int, extractor_comp_index: int, time: float, length: int = 40
    ):
        """Mock SpikeCutout object

        Parameters
        ----------
        spike_type : int
            0 for flat spike cutout
            1 for sine
            2 for triangle
        extractor_comp_index : int
        time : float
        length : int, default = 40
            number of sampling points
        """

        cutout: np.ndarray = np.zeros(length)

        if spike_type == 1:
            for index, point in enumerate(cutout):
                cutout[index] = np.sin(index * 2 * np.pi / length)

        if spike_type == 2:
            for index, point in enumerate(cutout):
                if index < length / 4:
                    cutout[index] = index * 4 / length
                elif index < 3 * length / 4:
                    cutout[index] = 2 - index * 4 / length
                else:
                    cutout[index] = -4 + index * 4 / length

        super().__init__(cutout, 30000, extractor_comp_index, time)


def test_len():
    cutout = MockSpikeCutout(0, 0, 0, 123)
    assert cutout.__len__() == 123


def test_get_cutouts_by_components():
    cutouts = []
    cutouts.append(MockSpikeCutout(0, 0, 0))
    cutouts.append(MockSpikeCutout(1, 1, 0.1))
    cutouts.append(MockSpikeCutout(2, 2, 0.2))
    cutouts.append(MockSpikeCutout(0, 0, 0.3))
    cutouts.append(MockSpikeCutout(1, 1, 0.4))
    cutouts.append(MockSpikeCutout(2, 2, 0.5))
    chan_spike_cutout = ChannelSpikeCutout(cutouts, 3, 0)

    cutouts_by_components = chan_spike_cutout.get_cutouts_by_component()
    assert np.shape(cutouts_by_components) == (3, 2)


def test_categorize():
    cutouts = []
    cutouts.append(MockSpikeCutout(0, 0, 0))
    cutouts.append(MockSpikeCutout(1, 1, 0.1))
    cutouts.append(MockSpikeCutout(2, 2, 0.2))
    cutouts.append(MockSpikeCutout(0, 0, 0.3))
    cutouts.append(MockSpikeCutout(1, 1, 0.4))
    cutouts.append(MockSpikeCutout(2, 2, 0.5))
    chan_spike_cutout = ChannelSpikeCutout(cutouts, 3, 0)

    cat_list = [2, 1, 0]
    chan_spike_cutout.categorize(cat_list)
    for spike_index, spike_cutout in enumerate(chan_spike_cutout.cutouts):
        assert spike_cutout.categorized
        assert (
            chan_spike_cutout.categorization_list[spike_cutout.extractor_comp_index]
            == cat_list[spike_cutout.extractor_comp_index]
        )


def test_labeled_cutouts():
    cutouts = []
    cutouts.append(MockSpikeCutout(0, 0, 0, 100))
    cutouts.append(MockSpikeCutout(1, 1, 0.1, 100))
    cutouts.append(MockSpikeCutout(2, 2, 0.2, 100))
    cutouts.append(MockSpikeCutout(0, 0, 0.3, 100))
    cutouts.append(MockSpikeCutout(1, 1, 0.4, 100))
    cutouts.append(MockSpikeCutout(2, 2, 0.5, 100))
    chan_spike_cutout = ChannelSpikeCutout(cutouts, 3, 0)

    cat_list = [-1, 1, 0]
    chan_spike_cutout.categorize(cat_list)
    labeled_cutouts = chan_spike_cutout.get_labeled_cutouts()
    assert np.shape(labeled_cutouts["labels"]) == (4,)
    assert np.shape(labeled_cutouts["labeled_cutouts"]) == (4, 100)
    assert labeled_cutouts["size"] == 4
