from typing import Iterable, Optional, Set

import enum
import json
import os
from contextlib import contextmanager
from enum import Enum

import numpy as np
import pytest

from miv.io.data import Data, DataManager
from tests.spike.cutout.test_cutout import MockSpikeCutout


class SignalGeneratorType(Enum):
    RANDOM = enum.auto()
    ARANGE = enum.auto()
    ONES = enum.auto()


class MockDataGenerator:
    @staticmethod
    def save_continuous_data_file(
        dirname,
        num_channels,
        signal_length,
        signaltype: SignalGeneratorType = SignalGeneratorType.RANDOM,
        seed=0,
    ):
        np_rng = np.random.default_rng(seed)
        if signaltype == SignalGeneratorType.RANDOM:
            signal = np_rng.random([signal_length, num_channels])
        elif signaltype == SignalGeneratorType.ARANGE:
            signal = np.arange(signal_length * num_channels).reshape(
                [signal_length, num_channels]
            )
        elif signaltype == SignalGeneratorType.ONES:
            signal = np.ones([signal_length, num_channels])
        else:
            raise NotImplementedError
        signal = signal.astype("int16")
        filename = os.path.join(dirname, "continuous.dat")
        fp = np.memmap(filename, dtype="int16", mode="w+", shape=signal.shape)
        fp[:] = signal[:]
        fp.flush()

        return signal

    # staticmethod
    def create_mock_data_structure(dirname, num_channels, signal_length):
        # Create mock datafile structure with continuous.dat, timestamps.np, and structure.oebin files.
        # Returns directory name and expected data values.

        # Prepare continuous.dat
        signal = MockDataGenerator.save_continuous_data_file(
            dirname, num_channels, signal_length
        )
        # Prepare timestamps.npy
        timestamps_filename = os.path.join(dirname, "timestamps.npy")
        timestamps = np.arange(signal_length) + np.pi
        np.save(timestamps_filename, timestamps)
        # Prepare structure.oebin
        oebin_filename = os.path.join(dirname, "structure.oebin")
        oebin = """{
        "continuous": [
            {
                "sample_rate": 30000,
                "num_channels": 3,
                "channels": [
                    {
                        "bit_volts":5.0,
                        "units":"uV",
                        "channel_name":"DC"
                    },
                    {
                        "bit_volts":3.0,
                        "units":"uV",
                        "channel_name":"DC_"
                    },
                    {
                        "bit_volts":2.5,
                        "units":"uV",
                        "channel_name":"DC_"
                    }
                ]
            }
        ]
    }"""
        with open(oebin_filename, "w") as f:
            f.write(oebin)

        # Expected Dataset
        expected_data = signal.copy().astype("float32")
        expected_data[:, 0] *= 5.0
        expected_data[:, 1] *= 3.0
        expected_data[:, 2] *= 2.5

        # Expected timestamps
        sampling_rate = 30000
        expected_timestamps = timestamps / sampling_rate

        return dirname, expected_data, expected_timestamps, sampling_rate


@pytest.fixture(name="create_mock_data_file")
def fixture_create_mock_data_file(tmp_path):
    return MockDataGenerator.create_mock_data_structure(
        tmp_path, num_channels=3, signal_length=100
    )


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


class AdvancedMockData(Data):
    def __init__(self):
        self.masking_channel_set: Set[int] = set()

        self.sampling_rate = 30000
        self.times = np.linspace(start=0, stop=self.sampling_rate * 70, num=70)

        self.signal = np.ndarray((6, 70))
        self.signal[0] = 1000 * MockSpikeCutout(0, 0, 0, length=70).cutout
        self.signal[1] = 1000 * MockSpikeCutout(1, 1, 0, length=70).cutout
        self.signal[2] = 1000 * MockSpikeCutout(2, 2, 0, length=70).cutout
        self.signal[3] = 1000 * MockSpikeCutout(0, 0, 0, length=70).cutout
        self.signal[4] = 1000 * MockSpikeCutout(1, 1, 0, length=70).cutout
        self.signal[5] = 1000 * MockSpikeCutout(2, 2, 0, length=70).cutout

    @contextmanager
    def load(self):
        yield np.transpose(self.signal), self.times, self.sampling_rate


# This MockSpontaneousData class generates signals with 5 channels that all look like
# channel 3 (all 100's) from MockData
class MockSpontaneousData(Data):
    def __init__(self):
        self.masking_channel_set: Set[int] = set()

        self.duration: float = 1
        self.sampling_rate = 10000
        self.times = np.arange(start=0, stop=self.duration, step=1 / self.sampling_rate)
        self.signal = []

        ones_signal = np.ones(self.duration * self.sampling_rate)
        for i in range(6):
            self.signal.append(100 * ones_signal)

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

    def auto_channel_mask_with_firing_rate(
        self, filter, detector, no_spike_threshold: float = 1
    ):
        super().auto_channel_mask_with_firing_rate(filter, detector, no_spike_threshold)


@pytest.fixture
def test_TTL_data(tmp_path):
    # Set up test data
    folder = tmp_path / "ttl_events"
    expected_states = [1, 0, 1, 0, 1]
    expected_full_words = [0, 1, 2, 3, 4]
    expected_timestamps = [0.0, 1.0, 2.0, 3.0, 4.0]
    expected_sampling_rate = 1000.0
    expected_initial_state = 0
    expected_sample_numbers = [0, 1000, 2000, 3000, 4000]

    # Create the test data files
    os.makedirs(folder, exist_ok=True)
    os.makedirs(os.path.join(folder, "events"), exist_ok=True)
    os.makedirs(os.path.join(folder, "events", "TTL_1"), exist_ok=True)
    np.save(os.path.join(folder, "events", "TTL_1", "states.npy"), expected_states)
    np.save(
        os.path.join(folder, "events", "TTL_1", "full_words.npy"), expected_full_words
    )
    np.save(
        os.path.join(folder, "events", "TTL_1", "timestamps.npy"), expected_timestamps
    )
    np.save(
        os.path.join(folder, "events", "TTL_1", "sample_numbers.npy"),
        expected_sample_numbers,
    )
    info = {
        "GUI version": "0.6.0",
        "events": [
            {
                "channel_name": "TTL Input",
                "folder_name": "TTL_1",
                "sample_rate": expected_sampling_rate,
                "initial_state": expected_initial_state,
            }
        ],
    }
    with open(os.path.join(folder, "structure.oebin"), "w") as f:
        json.dump(info, f)

    return (
        folder,
        expected_states,
        expected_full_words,
        expected_timestamps,
        expected_sampling_rate,
        expected_initial_state,
        expected_sample_numbers,
    )
