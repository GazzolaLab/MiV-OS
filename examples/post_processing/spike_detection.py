import os

import matplotlib.pyplot as plt
import neo
import numpy as np
import quantities as pq
from viziphant.rasterplot import rasterplot_rates

<<<<<<< HEAD
from miv.io import load_data
from miv.signal.filter import butter_bandpass_filter
from miv.signal.spike import (
    align_to_minimum,
    compute_spike_threshold,
    detect_threshold_crossings,
)
||||||| parent of 6ae1418 (update spike detection example script)
from miv.io import load_data
from miv.signal.filter import butter_bandpass_filter
from miv.signal.spike import (
    compute_spike_threshold,
    detect_threshold_crossings,
    align_to_minimum,
)
=======
from miv.io import Data, DataManager
from miv.signal.filter import ButterBandpass
from miv.signal.spike import ThresholdCutoff
>>>>>>> 6ae1418 (update spike detection example script)


def main():
    """Example spike detection script"""

    # Load dataset from OpenEphys recording
    folder_path: str = "~/Open Ephys/2022-03-10-16-19-09"  # Data Path
    data_manager = DataManager(folder_path)

    # Get signal and rate(hz)
    #   signal        : np.array, shape(N, N_channels)
    #   timestamps    : np.array
    #   sampling_rate : float
    with data_manager[0].load() as (signal, timestamps, sampling_rate):
        # Butter bandpass filter
        bandpass_filter = ButterBandpass(300, 3000, order=5)
        signal = bandpass_filter(signal, sampling_rate)

        # Spike Detection
        detector = ThresholdCutoff(cutoff=4.5)
        spiketrains = detector(signal, timestamps, sampling_rate)

    # Plot
    rasterplot_rates(spiketrains)


if __name__ == "__main__":
    main()
