from typing import Iterable, Optional, Set

from contextlib import contextmanager

import numpy as np

# from miv.typing import FilterProtocol, SpikeDetectionProtocol
from miv.io.data import Data, DataManager


class MockData(Data):
    def __init__(self):
        self.masking_channel_set: Set[int] = set()

        self.duration: float = 1
        self.sampling_rate = 10000
        self.times = np.arange(start=0, stop=self.duration, step=1 / self.sampling_rate)
        self.signal = []
        # Channels 0 and 1 give zeroes for signal
        self.signal.append(np.zeros(self.duration * self.sampling_rate))
        self.signal.append(np.zeros(self.duration * self.sampling_rate))
        # signal.append(np.random.rand(int(sampling_rate*duration)) * 30 - 15)
        self.signal = np.transpose(self.signal)

    @contextmanager
    def load(self):
        yield self.signal, self.times, self.sampling_rate


class MockDataManager(DataManager):
    def __init__(self, mock_data: Optional[Data] = None):
        self.data_list = []
        data = mock_data if mock_data else MockData()
        for i in range(2):
            self.data_list.append(data)

    def auto_channel_mask_baseline(
        self, filter, detector, no_spike_threshold: float = 1
    ):
        super().auto_channel_mask_baseline(filter, detector, no_spike_threshold)
