from typing import Any, Iterable

import numpy as np
from miv.signal.spike import SpikeDetectionProtocol


class SpikeDetection1:
    def __call__(self, a):
        return a


mock_spike_detection_list: Iterable[SpikeDetectionProtocol] = [SpikeDetection1]


class NonSpikeDetection1:
    pass


mock_nonspike_detection_list: Iterable[Any] = [NonSpikeDetection1]
