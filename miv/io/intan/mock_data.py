"""Utilities to create mock Intan RHS data files for testing."""

import os
import struct
import xml.etree.ElementTree as ET
from typing import BinaryIO

import numpy as np


def write_qstring(fid: BinaryIO, s: str) -> None:
    """Write Qt style QString.

    The first 32-bit unsigned number indicates the length of the string (in bytes).
    Strings are stored as unicode.
    """
    if not s:
        fid.write(struct.pack("<I", 0xFFFFFFFF))
        return

    # Convert string to unicode bytes (UTF-16LE)
    utf16_bytes = s.encode("utf-16-le")
    length = len(utf16_bytes)
    fid.write(struct.pack("<I", length))
    fid.write(utf16_bytes)


def create_mock_rhs_file(
    filepath: str,
    num_channels: int,
    sampling_rate: float,
    duration_seconds: float,
    signal_data: np.ndarray | None = None,
) -> None:
    """Create a mock RHS file with the specified parameters.

    Parameters
    ----------
    filepath : str
        Path where the RHS file will be created
    num_channels : int
        Number of amplifier channels
    sampling_rate : float
        Sampling rate in Hz
    duration_seconds : float
        Duration of the recording in seconds
    signal_data : np.ndarray, optional
        Optional signal data. If None, generates random data.
        Shape should be (num_samples, num_channels)
    """
    num_samples = int(sampling_rate * duration_seconds)
    num_data_blocks = (num_samples + 127) // 128  # Round up to nearest 128-sample block

    # Calculate bytes per data block
    # 128 samples * 4 bytes (timestamps) + 128 * 2 * num_channels (amplifier) + 128 * 2 * num_channels (stim)
    bytes_per_block = 128 * 4 + 128 * 2 * num_channels + 128 * 2 * num_channels

    # We'll calculate header_size after writing it
    header_size = None

    with open(filepath, "wb") as fid:
        header_start = fid.tell()
        # Write magic number
        fid.write(struct.pack("<I", 0xD69127AC))

        # Write version number
        fid.write(struct.pack("<hh", 1, 0))  # Version 1.0

        # Write sample rate
        fid.write(struct.pack("<f", sampling_rate))

        # Write DSP and bandwidth settings (all zeros for simplicity)
        fid.write(struct.pack("<hffffffff", 0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0))

        # Write notch filter mode (0 = disabled)
        fid.write(struct.pack("<h", 0))

        # Write impedance test frequencies
        fid.write(struct.pack("<ff", 0.0, 0.0))

        # Write amp settle and charge recovery modes
        fid.write(struct.pack("<hh", 0, 0))

        # Write stimulation parameters
        fid.write(struct.pack("fff", 0.0, 0.0, 0.0))

        # Write notes (empty strings)
        write_qstring(fid, "")
        write_qstring(fid, "")
        write_qstring(fid, "")

        # Write DC amplifier data saved flag and eval board mode
        fid.write(struct.pack("<hh", 0, 0))

        # Write reference channel (empty)
        write_qstring(fid, "")

        # Write signal groups (1 group with num_channels channels)
        fid.write(struct.pack("<h", 1))  # number_of_signal_groups

        # Write signal group info
        write_qstring(fid, "Port A")
        write_qstring(fid, "A")
        fid.write(
            struct.pack("<hhh", 1, num_channels, num_channels)
        )  # enabled, num_channels, num_amp_channels

        # Write channel info for each channel
        for ch in range(num_channels):
            write_qstring(fid, f"A-{ch:03d}")
            write_qstring(fid, f"CH{ch}")
            fid.write(
                struct.pack("<hhhhhhh", ch, ch, 0, 1, ch, 0, 0)
            )  # native_order, custom_order, signal_type, enabled, chip_channel, command_stream, board_stream
            # Write trigger channel info
            fid.write(struct.pack("<hhhh", 0, 0, 0, 0))
            # Write impedance
            fid.write(struct.pack("<ff", 0.0, 0.0))

        # Write spike triggers (empty)
        fid.write(struct.pack("<h", 0))

        # Write board ADC channels (none)
        fid.write(struct.pack("<h", 0))

        # Write board DAC channels (none)
        fid.write(struct.pack("<h", 0))

        # Write board digital input channels (none)
        fid.write(struct.pack("<h", 0))

        # Write board digital output channels (none)
        fid.write(struct.pack("<h", 0))

        header_end = fid.tell()
        header_size = header_end - header_start

        # Now write data blocks
        if signal_data is None:
            # Generate random signal data
            signal_data = np.random.randint(
                0, 65535, size=(num_samples, num_channels), dtype=np.uint16
            )
        else:
            # Ensure signal_data has correct shape and type
            if signal_data.shape[0] < num_samples:
                # Pad with zeros if needed
                padding = np.zeros(
                    (num_samples - signal_data.shape[0], num_channels), dtype=np.uint16
                )
                signal_data = np.vstack([signal_data, padding])
            elif signal_data.shape[0] > num_samples:
                signal_data = signal_data[:num_samples, :]
            signal_data = signal_data.astype(np.uint16)

        # Calculate actual samples to write (must be multiple of 128)
        actual_samples = num_data_blocks * 128
        if signal_data.shape[0] < actual_samples:
            padding = np.zeros(
                (actual_samples - signal_data.shape[0], num_channels), dtype=np.uint16
            )
            signal_data = np.vstack([signal_data, padding])

        # Write data blocks
        for block_idx in range(num_data_blocks):
            start_idx = block_idx * 128
            end_idx = start_idx + 128

            # Write timestamps (sample indices)
            timestamps = np.arange(start_idx, end_idx, dtype=np.int32)
            fid.write(timestamps.tobytes())

            # Write amplifier data (uint16, channels x samples)
            block_data = signal_data[
                start_idx:end_idx, :
            ].T  # Transpose to (channels, samples)
            fid.write(block_data.tobytes())

            # Write DC amplifier data (not saved, but space reserved)
            # Skip since dc_amplifier_data_saved = 0

            # Write stimulation data (uint16, channels x samples, all zeros)
            stim_data = np.zeros((num_channels, 128), dtype=np.uint16)
            fid.write(stim_data.tobytes())

    # Verify and fix file size using the actual RHS reader's calculation
    # Read the header to get the actual bytes_per_block calculation
    from miv.io.intan import rhs

    with open(filepath, "rb") as fid:
        try:
            read_header = rhs.read_header(fid)
            header_end_pos = fid.tell()
            calculated_bytes_per_block = rhs.get_bytes_per_data_block(read_header)
            actual_file_size = os.path.getsize(filepath)
            # bytes_remaining = actual_file_size - header_end_pos

            # Calculate expected file size based on reader's calculation
            expected_bytes_remaining = num_data_blocks * calculated_bytes_per_block
            expected_file_size = header_end_pos + expected_bytes_remaining

            if actual_file_size != expected_file_size:
                # Fix file size
                with open(filepath, "r+b") as fix_fid:
                    if actual_file_size > expected_file_size:
                        fix_fid.truncate(expected_file_size)
                    elif actual_file_size < expected_file_size:
                        fix_fid.seek(0, 2)  # Seek to end
                        padding_needed = expected_file_size - actual_file_size
                        fix_fid.write(b"\x00" * padding_needed)
        except Exception:
            # If reading fails, fall back to our calculation
            actual_file_size = os.path.getsize(filepath)
            expected_file_size = header_size + num_data_blocks * bytes_per_block
            if actual_file_size != expected_file_size:
                with open(filepath, "r+b") as fix_fid:
                    if actual_file_size > expected_file_size:
                        fix_fid.truncate(expected_file_size)
                    elif actual_file_size < expected_file_size:
                        fix_fid.seek(0, 2)
                        padding_needed = expected_file_size - actual_file_size
                        fix_fid.write(b"\x00" * padding_needed)


def create_settings_xml(filepath: str, sampling_rate: int, num_channels: int) -> None:
    """Create a settings.xml file for Intan data.

    Parameters
    ----------
    filepath : str
        Path where settings.xml will be created
    sampling_rate : int
        Sampling rate in Hz
    num_channels : int
        Number of channels
    """
    root = ET.Element("Settings", SampleRateHertz=str(sampling_rate))

    # Create SignalGroup for port A
    signal_group = ET.SubElement(root, "SignalGroup", Prefix="A")

    # Create channels
    for _ in range(num_channels):
        ET.SubElement(signal_group, "Channel", Enabled="True")

    # Write XML to file
    tree = ET.ElementTree(root)
    ET.indent(tree, space="  ")
    tree.write(filepath, encoding="utf-8", xml_declaration=True)


def create_mock_rhs_data_folder(
    output_dir: str,
    num_files: int = 1,
    num_channels: int = 4,
    sampling_rate: int = 30000,
    duration_seconds: float = 10.0,
    folder_name: str | None = None,
) -> str:
    """Create a folder with mock RHS data files and settings.xml.

    Parameters
    ----------
    output_dir : str
        Directory where the mock data folder will be created
    num_files : int
        Number of RHS files to create (default: 1)
    num_channels : int
        Number of channels per file (default: 4)
    sampling_rate : int
        Sampling rate in Hz (default: 30000)
    duration_seconds : float
        Duration of each file in seconds (default: 10.0)
    folder_name : str, optional
        Name of the folder. If None, uses "mock_rhs_data"

    Returns
    -------
    str
        Path to the created folder
    """
    if folder_name is None:
        folder_name = "mock_rhs_data"

    folder_path = os.path.join(output_dir, folder_name)
    os.makedirs(folder_path, exist_ok=True)

    # Create settings.xml
    settings_path = os.path.join(folder_path, "settings.xml")
    create_settings_xml(settings_path, sampling_rate, num_channels)

    # Create RHS files
    for i in range(num_files):
        rhs_filename = f"data_000{i:03d}.rhs"
        rhs_path = os.path.join(folder_path, rhs_filename)
        create_mock_rhs_file(
            rhs_path,
            num_channels=num_channels,
            sampling_rate=sampling_rate,
            duration_seconds=duration_seconds,
        )

    return folder_path
