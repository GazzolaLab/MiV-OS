import numpy as np
import tensorflow as tf

from miv.signal.classification.abnormality_detection import AbnormalityDetector
from miv.signal.filter.butter_bandpass_filter import ButterBandpass
from miv.signal.spike.detection import ThresholdCutoff
from miv.signal.spike.sorting import PCADecomposition
from tests.io.mock_data import AdvancedMockData


class MockAbnormalityDetector(AbnormalityDetector):
    def __init__(self) -> None:
        # 6 channels, length = 70
        mock_data = AdvancedMockData()
        AbnormalityDetector.spontaneous_data = mock_data
        AbnormalityDetector.spontaneous_cutouts = mock_data.signal
        AbnormalityDetector.categorized = False
        self.filter = ButterBandpass(lowcut=300, highcut=3000)
        self.spike_detector = ThresholdCutoff()
        self.extractor = PCADecomposition()


def test_get_all_cutouts():
    abn_detector = MockAbnormalityDetector()
    all_cutouts = abn_detector._get_all_cutouts(
        abn_detector.spontaneous_data,
        signal_filter=abn_detector.filter,
        spike_detector=abn_detector.spike_detector,
        extractor=abn_detector.extractor,
        extractor_decomp_param=3,
    )

    assert np.shape(all_cutouts)[0] == 6
