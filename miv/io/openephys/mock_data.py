"""Utilities to create mock OpenEphys data files for testing."""

import os
import json

import numpy as np


def create_recording_data(
    output_dir: str,
    num_channels: int,
    sampling_rate: float = 30000.0,
    duration_seconds: float = 10.0,
    processor_name: str = "Rhythm_FPGA-100.0",
) -> tuple[str, np.ndarray, np.ndarray, float]:
    """Create arbitrary N-channel recording data with specified frequency for T-seconds.

    The format is compatible with OpenEphys recording data and loadable with
    miv.io.openephys.Data.

    Parameters
    ----------
    output_dir : str
        Directory where the recording data will be created
    num_channels : int
        Number of channels in the recording
    sampling_rate : float
        Sampling rate in Hz (default: 30000.0)
    duration_seconds : float
        Duration of the recording in seconds (default: 10.0)
    processor_name : str
        Name of the processor folder (default: "Rhythm_FPGA-100.0")

    Returns
    -------
    tuple
        A tuple containing:
        - data_path (str): Path to the created recording directory
        - data (np.ndarray): The signal data after bit_volts conversion (float32)
        - timestamps (np.ndarray): The timestamps array
        - sampling_rate (float): The sampling rate
    """
    # Create directory structure
    # If output_dir already ends with "recording", use it directly; otherwise create "recording1" subdirectory
    if os.path.basename(output_dir).startswith("recording"):
        recording_dir = output_dir
    else:
        recording_dir = os.path.join(output_dir, "recording1")
    continuous_dir = os.path.join(recording_dir, "continuous", processor_name)
    os.makedirs(continuous_dir, exist_ok=True)

    # Calculate number of samples
    num_samples = int(sampling_rate * duration_seconds)

    # Create continuous.dat file (int16 format)
    continuous_dat_path = os.path.join(continuous_dir, "continuous.dat")
    signal_data = np.random.randint(
        -32768, 32767, size=(num_samples, num_channels), dtype=np.int16
    )
    signal_data.tofile(continuous_dat_path)

    # Create timestamps.npy file
    timestamps_path = os.path.join(continuous_dir, "timestamps.npy")
    timestamps = np.arange(num_samples, dtype=np.float64) / sampling_rate
    np.save(timestamps_path, timestamps)

    # Create structure.oebin file
    oebin_path = os.path.join(recording_dir, "structure.oebin")
    channels = []
    bit_volts = 1.0
    for i in range(num_channels):
        channels.append(
            {"bit_volts": bit_volts, "units": "uV", "channel_name": f"CH{i}"}
        )

    oebin_data = {
        "continuous": [
            {
                "sample_rate": sampling_rate,
                "num_channels": num_channels,
                "channels": channels,
            }
        ]
    }

    with open(oebin_path, "w") as f:
        json.dump(oebin_data, f, indent=2)

    # Convert signal data to voltage (matching bits_to_voltage conversion)
    # Convert int16 to float32 and apply bit_volts conversion
    converted_data = signal_data.astype(np.float32) * bit_volts

    return recording_dir, converted_data, timestamps, sampling_rate


def create_dataset(
    output_dir: str,
    num_record_nodes: int = 1,
    num_experiments: int = 1,
    num_recordings_per_experiment: int = 1,
    num_channels: int = 4,
    sampling_rate: float = 30000.0,
    duration_seconds: float = 10.0,
    processor_name: str = "Rhythm_FPGA-100.0",
) -> str:
    """Create multiple recording data that is loadable with miv.io.openephys.DataManager.

    Creates a dataset with the structure expected by DataManager:
    data_collection_path/
      Record Node X/
        experimentY/
          recordingZ/

    Parameters
    ----------
    output_dir : str
        Directory where the dataset will be created
    num_record_nodes : int
        Number of record nodes (default: 1)
    num_experiments : int
        Number of experiments per record node (default: 1)
    num_recordings_per_experiment : int
        Number of recordings per experiment (default: 1)
    num_channels : int
        Number of channels in each recording (default: 4)
    sampling_rate : float
        Sampling rate in Hz (default: 30000.0)
    duration_seconds : float
        Duration of each recording in seconds (default: 10.0)
    processor_name : str
        Name of the processor folder (default: "Rhythm_FPGA-100.0")

    Returns
    -------
    str
        Path to the created data collection directory
    """
    # Create data collection directory
    data_collection_path = os.path.join(output_dir, "data_collection")
    os.makedirs(data_collection_path, exist_ok=True)

    # Create recordings for each record node, experiment, and recording
    for record_node_idx in range(num_record_nodes):
        record_node_name = f"Record Node {record_node_idx + 1}"
        for exp_idx in range(num_experiments):
            experiment_name = f"experiment{exp_idx}"
            for rec_idx in range(num_recordings_per_experiment):
                recording_name = f"recording{rec_idx}"

                # Create the recording path structure
                recording_path = os.path.join(
                    data_collection_path,
                    record_node_name,
                    experiment_name,
                    recording_name,
                )

                # Use create_recording_data to create the actual recording
                create_recording_data(
                    output_dir=recording_path,
                    num_channels=num_channels,
                    sampling_rate=sampling_rate,
                    duration_seconds=duration_seconds,
                    processor_name=processor_name,
                )

    return data_collection_path
