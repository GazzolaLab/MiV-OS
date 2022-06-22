from typing import Iterable, Set

from contextlib import contextmanager

import numpy as np

from miv.io.data import Data, DataManager

# class MockData(Data):
#     def __init__(self):
#         self.masking_channel_set: Set[int] = set()

#     @contextmanager
#     def load(self):
#         num_channels = 64
#         duration: float = 1.5
#         sampling_rate = 10000

#         signal = np.random.rand(int(sampling_rate*duration), num_channels) * 30 - 15
#         times = np.arange(start=0, stop=duration, step=1/sampling_rate)

#         yield signal, times, sampling_rate

#     def set_channel_mask(self, channel_id: Iterable[int]):
#         self.masking_channel_set.update(channel_id)


class MockDataManager(DataManager):
    def __init__(self, mock_data):
        self.data_list = []
        for i in range(2):
            self.data_list.append(mock_data())
