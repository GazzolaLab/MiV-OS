import numpy as np
import tensorflow as tf

from miv.signal.classification.abnormality_detection import DetectorWithSpontaneousData
from miv.signal.filter.butter_bandpass_filter import ButterBandpass
from miv.signal.spike.cutout import ChannelSpikeCutout, SpikeCutout
from miv.signal.spike.detection import ThresholdCutoff
from miv.signal.spike.sorting import PCADecomposition
from tests.io.mock_data import AdvancedMockData


class MockDetectorWithSpontaneousData(DetectorWithSpontaneousData):
    def __init__(self) -> None:
        # 2 channels, 6 cutouts each, 3 components, length = 70
        # [[0, 1, 2, 0, 1, 2], [0, 1, 2, 0, 1, 2]]
        mock_data = AdvancedMockData()
        DetectorWithSpontaneousData.spontaneous_data = mock_data

        cutouts: np.ndarray = np.ndarray(6, dtype=SpikeCutout)
        for i in range(6):
            cutouts[i] = SpikeCutout(mock_data.signal[i], 30000, i % 3, 0)

        all_cutouts: np.ndarray = np.array(
            [ChannelSpikeCutout(cutouts, 3, 0), ChannelSpikeCutout(cutouts, 3, 1)]
        )

        DetectorWithSpontaneousData.spontaneous_cutouts = all_cutouts
        DetectorWithSpontaneousData.categorized = False
        self.filter = ButterBandpass(lowcut=300, highcut=3000)
        self.spike_detector = ThresholdCutoff()
        self.extractor = PCADecomposition()


class TestDetectorWithSpontaneousData:
    def test_get_all_cutouts(self):
        abn_detector = MockDetectorWithSpontaneousData()
        all_cutouts = abn_detector._get_all_cutouts(
            abn_detector.spontaneous_data,
            signal_filter=abn_detector.filter,
            spike_detector=abn_detector.spike_detector,
            extractor=abn_detector.extractor,
            extractor_decomp_param=3,
        )

        assert np.shape(all_cutouts)[0] == 6

    def test_categorize_spontaneous(self):
        abn_detector = MockDetectorWithSpontaneousData()

        # Case 1: decomp param does not match
        cat = np.array([[0, 0, 0, 1], [1, 0, 0, 1]])
        try:
            abn_detector.categorize_spontaneous(cat)
        except IndexError:
            pass

        # Case 2: num channels does not match
        cat = np.array([0, 0, 1, 0, 1, 1])
        try:
            abn_detector.categorize_spontaneous(cat)
        except IndexError:
            pass

        # Case 3: normal categorization
        cat = np.array([[0, 1, 1], [0, 1, -1]])
        abn_detector.categorize_spontaneous(cat)
        chan0 = abn_detector.spontaneous_cutouts[0]
        chan1 = abn_detector.spontaneous_cutouts[1]
        assert abn_detector.categorized
        assert chan0.categorized
        assert not chan1.categorized
        assert np.array_equal(chan0.categorization_list, np.array([0, 1, 1]))
        assert np.array_equal(chan1.categorization_list, np.array([0, 1, -1]))

    def test_evaluate_model(self):
        abn_detector = MockDetectorWithSpontaneousData()
        abn_detector.categorize_spontaneous([[0, 1, 1], [0, 1, 1]])
        abn_detector.default_init_and_compile_classifier()
        abn_detector.default_init_and_train_model()
        test_spikes = abn_detector.spontaneous_cutouts[0].get_labeled_cutouts()[
            "cutouts"
        ]

        test_labels = np.array([0, 1, 1, 0, 1, 1])
        metrics = abn_detector.evaluate_model(test_spikes, test_labels)
        assert metrics["accuracy"] == 1
        assert metrics["f1"] == 1
