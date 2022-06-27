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
        zero_signal = np.zeros(self.duration * self.sampling_rate)
        self.signal.append(zero_signal)
        self.signal.append(zero_signal)

        # Channel 2 gives ones for signal
        ones_signal = np.ones(self.duration * self.sampling_rate)
        self.signal.append(ones_signal)

        # Channel 3 gives 100's for signal
        self.signal.append(100 * ones_signal)

        # Channel 4 has 10 spikes
        ten_spikes_signal = zero_signal.copy()
        for i in range(999, 10000, 1000):
            ten_spikes_signal[i] = 1000
            ten_spikes_signal[i - 1] = 1000
            ten_spikes_signal[i - 2] = 1000
        self.signal.append(ten_spikes_signal)

        # Channel 5 has 3 spikes in the first 0.5 second
        channel_5 = zero_signal.copy()
        for i in range(800, 5800, 2000):
            channel_5[i] = 1000
        self.signal.append(channel_5)

        self.signal = np.transpose(self.signal)

    @contextmanager
    def load(self):
        yield self.signal, self.times, self.sampling_rate


class MockDataManager(DataManager):
    def __init__(self, mock_data: Optional[Data] = None):
        self.data_list = []

        if mock_data:
            self.data_list.append(mock_data)
            self.data_list.append(mock_data)

        else:
            self.data_list.append(MockData())
            self.data_list.append(MockData())

    def auto_channel_mask_baseline(
        self, filter, detector, no_spike_threshold: float = 1
    ):
        super().auto_channel_mask_baseline(filter, detector, no_spike_threshold)
