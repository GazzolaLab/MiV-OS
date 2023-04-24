__doc__ = """
Module for filtering spiketrain based on waveform statistical properties.

Motivated by _[1].

[1]. Toosi R., et al. An automatic spike sorting algorithm based on adaptive spike detection and a mixture of skew-t distributions (2021).
"""
__all__ = ["WaveformStatisticalFilter"]

from typing import Dict, Optional

import csv
import os
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np

from miv.core.datatype import Signal, Spikestamps
from miv.core.operator import OperatorMixin


@dataclass
class WaveformStatisticalFilter(OperatorMixin):
    """
    Filter spike train based on waveform statistical properties.

    Mainly, the waveform that has a mean larger than 1.0 and standard deviation larger than 3.0.

    .. code::

        >>> filtered_signal = ButterBandpass(300, 3000, order=4)
        >>> spiketrains = ThresholdCutoff(cutoff=4.0)
        >>> waveform = ExtractWaveforms()

        >>> filtered_signal >> spiketrains
        >>> filtered_signal >> waveform; spiketrains >> waveform
        >>> waveform >> statistical_filter; spiketrains >> statistical_filter

        >>> Pipeline(statistical_filter).run()

    Parameters
    ----------
    max_mean : float
        The maximum mean of the waveform. (default: 1.0)
    max_std : float
        The maximum standard deviation of the waveform. (default: 3.0)
    """

    max_mean: float = 1.0
    max_std: float = 10.0

    tag: str = "waveform statistical filter"

    def __post_init__(self):
        super().__init__()

    def __call__(
        self, waveforms: Dict[int, Signal], spiketrains: Spikestamps
    ) -> Spikestamps:
        """
        Parameters
        ----------
        waveforms : Dict[int, Signal]
            The waveforms of each channel.
        spiketrains : Spikestamps
            The spiketrains of each channel.

        Returns
        -------
        Spikestamps
            The filtered spiketrains of each channel.
        """

        filtered_spiketrains = Spikestamps()
        total_spikecounts = spiketrains.get_count()
        log = []
        for channel in range(spiketrains.number_of_channels):
            if channel not in waveforms:
                filtered_spiketrains.append(np.array([]))
                continue
            waveform = waveforms[channel]
            spiketrain = spiketrains[channel]
            total_spikecount = total_spikecounts[channel]

            spike_mean = np.mean(waveform.data, axis=waveform._SIGNALAXIS)
            spike_std = np.std(waveform.data, axis=waveform._SIGNALAXIS)

            indices = (
                (spike_std > self.max_std)
                | (spike_mean > self.max_mean)
                | (spike_mean < -self.max_mean)
            )
            number_of_outliers = indices.sum()

            filtered_spiketrains.append(np.asarray(spiketrain)[~indices])
            log.append(
                (
                    channel,
                    total_spikecount,
                    number_of_outliers,
                    number_of_outliers / total_spikecount,
                )
            )

        self._log = log  # FIXME: Maybe better way to pass down the result

        return filtered_spiketrains

    def after_run_save_log(self, *args, **kwargs):
        log = self._log
        # Save log in csv
        savepath = os.path.join(self.analysis_path, "filtered_statistics.csv")
        with open(savepath, "w", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(
                ["channel", "total spikecount", "number of filtered outliers", "ratio"]
            )
            for row in log:
                writer.writerow(row)
