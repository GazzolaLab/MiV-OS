import pytest

import tempfile
import os
import numpy as np

from miv.io.binary import (
    load_continuous_data,
    load_recording,
    oebin_read,
    apply_channel_mask,
)


@pytest.mark.parametrize("signal", [np.arange(10), np.array([])])
def test_apply_channel_mask_shape_failure(signal):
    with pytest.raises(IndexError):
        apply_channel_mask(signal, {0})


@pytest.mark.parametrize("signal", [5, [1, 2, 3], (1, 5, 9)])
def test_apply_channel_mask_non_numpy_ndtype(signal):
    with pytest.raises(AttributeError):
        apply_channel_mask(signal, {0})


@pytest.mark.parametrize(
    "signal, mask, solution",
    [
        (np.arange(9).reshape([3, 3]), {0}, np.arange(9).reshape([3, 3])[:, [1, 2]]),
        (np.arange(9).reshape([3, 3]), {1}, np.arange(9).reshape([3, 3])[:, [0, 2]]),
        (np.arange(9).reshape([3, 3]), {0, 1}, np.arange(9).reshape([3, 3])[:, [2]]),
        (
            np.arange(9).reshape([3, 3]),
            {1, -1, 5},
            np.arange(9).reshape([3, 3])[:, [0, 2]],
        ),
    ],
)
def test_apply_channel_mask_functionality(signal, mask, solution):
    output = apply_channel_mask(signal, mask)
    np.testing.assert_allclose(output, solution)


def test_oebin_read_functionality():
    a = {"a": 1, "b": "string data", "c": True}
    with tempfile.NamedTemporaryFile("w+t", delete=False) as fp:
        fp.writelines(str(a))
        fp.seek(0)
        b = oebin_read(fp.name)
    assert a == b


@pytest.mark.parametrize("num_channels, signal_length", [(4, 100), (1, 50), (10, 5)])
def test_load_continuous_data_temp_file_without_timestamps(num_channels, signal_length):
    signal = np.arange(signal_length * num_channels).reshape(
        [signal_length, num_channels]
    )
    filename = os.path.join(tempfile.mkdtemp(), "continuous.dat")
    fp = np.memmap(filename, dtype="int16", mode="w+", shape=signal.shape)
    fp[:] = signal[:]
    fp.flush()

    raw_data, timestamps = load_continuous_data(fp.filename, num_channels, 1)
    np.testing.assert_allclose(timestamps, np.arange(signal_length))
    np.testing.assert_allclose(raw_data, signal)


@pytest.mark.parametrize(
    "num_channels, signal_length, freq", [(4, 100, 1), (1, 50, 5), (10, 5, 2)]
)
def test_load_continuous_data_temp_file_with_timestamps_shift(
    num_channels, signal_length, freq
):
    signal = np.arange(signal_length * num_channels).reshape(
        [signal_length, num_channels]
    )
    dirname = tempfile.mkdtemp()
    filename = os.path.join(dirname, "continuous.dat")
    timestamps_filename = os.path.join(dirname, "timestamps.npy")
    # Prepare continuous.dat
    fp = np.memmap(filename, dtype="int16", mode="w+", shape=signal.shape)
    fp[:] = signal[:]
    fp.flush()
    # Prepare timestamps.npy
    timestamps = np.arange(signal_length) + np.pi
    np.save(timestamps_filename, timestamps)

    # With shift
    raw_data, out_timestamps = load_continuous_data(
        fp.filename, num_channels, freq, start_at_zero=False
    )
    np.testing.assert_allclose(out_timestamps, timestamps / freq)
    np.testing.assert_allclose(raw_data, signal)

    # Without shift
    raw_data, out_timestamps = load_continuous_data(
        fp.filename, num_channels, freq, start_at_zero=True
    )
    np.testing.assert_allclose(out_timestamps, (timestamps - np.pi) / freq)
    np.testing.assert_allclose(raw_data, signal)


@pytest.mark.parametrize(
    "num_channels, signal_length, freq", [(4, 100, 1), (1, 50, 5), (10, 5, 2)]
)
def test_load_continuous_data_temp_file_timestamps_path_test(
    num_channels, signal_length, freq
):
    signal = np.arange(signal_length * num_channels).reshape(
        [signal_length, num_channels]
    )
    dirname = tempfile.mkdtemp()
    filename = os.path.join(dirname, "continuous.dat")
    timestamps_filename = os.path.join(dirname, "a.npy")
    # Prepare continuous.dat
    fp = np.memmap(filename, dtype="int16", mode="w+", shape=signal.shape)
    fp[:] = signal[:]
    fp.flush()
    # Prepare timestamps.npy
    timestamps = np.arange(signal_length)
    np.save(timestamps_filename, timestamps)

    # With shift
    raw_data, out_timestamps = load_continuous_data(
        fp.filename, num_channels, freq, "a.npy"
    )
    np.testing.assert_allclose(out_timestamps, timestamps / freq)
    np.testing.assert_allclose(raw_data, signal)


@pytest.mark.parametrize(
    "num_channels, signal_length, freq", [(4, 100, 1), (1, 50, 5), (10, 5, 2)]
)
def test_load_recording_assertion_single_data_file(num_channels, signal_length, freq):
    signal = np.arange(signal_length * num_channels).reshape(
        [signal_length, num_channels]
    )

    dirname = tempfile.mkdtemp()
    os.makedirs(os.path.join(dirname, "continuous", "temp1"))
    os.makedirs(os.path.join(dirname, "continuous", "temp2"))
    filename1 = os.path.join(dirname, "continuous", "temp1", "continuous.dat")
    filename2 = os.path.join(dirname, "continuous", "temp2", "continuous.dat")
    # Prepare continuous.dat
    fp1 = np.memmap(filename1, dtype="int16", mode="w+", shape=signal.shape)
    fp2 = np.memmap(filename2, dtype="int16", mode="w+", shape=signal.shape)
    fp1[:] = 1.0
    fp2[:] = 2.0
    fp1.flush()
    fp2.flush()

    with pytest.raises(
        AssertionError, match=r"(?=.*temp1.*)(?=.*temp2.*)(?=There should be only one)"
    ):
        load_recording(dirname)
