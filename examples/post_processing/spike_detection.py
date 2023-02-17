import os

import matplotlib.pyplot as plt
import neo
import numpy as np
import quantities as pq
from viziphant.rasterplot import rasterplot_rates

from miv.io import load_data
from miv.signal.filter import butter_bandpass_filter
from miv.signal.spike import (
    align_to_minimum,
    compute_spike_threshold,
    detect_threshold_crossings,
)


def main():
    """Example spike detection script"""

    # Load dataset from OpenEphys recording
    folder_path: str = "~/Open Ephys/2022-03-10-16-19-09"  # Data Path
    dataset = load_data(folder_path, device="OpenEphys")

    # Get signal and rate(hz)
    #   signal     : np.array, shape(N, N_channels)
    #   rate       : float
    record_node: int = dataset.get_nodes[0]
    recording = dataset[record_node]["experiment1"]["recording1"]
    signal, _, rate = recording.continuous["100"]
    # time = recording.continuous["100"].timestamp / rate
    num_channels = signal.shape[1]

    # Butter bandpass filter
    signal = butter_bandpass_filter(signal, lowcut=300, highcut=3000, fs=rate, order=5)

    # Spike detection for each channel
    spiketrain_list = []
    for channel in range(num_channels):
        # Spike Detection: get spikestamp
        spike_threshold = compute_spike_threshold(signal)
        crossings = detect_threshold_crossings(signal, rate, spike_threshold, 0.003)
        spikes = align_to_minimum(signal, rate, crossings, 0.002)
        spikestamp = spikes / rate
        # Convert spikestamp to neo.SpikeTrain (for plotting)
        spiketrain = neo.SpikeTrain(spikestamp, units="sec")
        spiketrain_list.append(spiketrain)

    # Plot
    rasterplot_rates(spiketrain_list)


if __name__ == "__main__":
    main()
