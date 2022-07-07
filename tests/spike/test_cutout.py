import numpy as np

from miv.signal.spike.cutout import ChannelSpikeCutout, SpikeCutout


class MockSpikeCutout(SpikeCutout):
    def __init__(
        self, spike_type: int, pca_comp_index: int, time: float, length: int = 40
    ):
        """Mock SpikeCutout object

        Parameters
        ----------
        spike_type : int
            0 for flat spike cutout
            1 for sine
            2 for triangle
        pca_comp_index : int
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

        super().__init__(cutout, 30000, pca_comp_index, time)
