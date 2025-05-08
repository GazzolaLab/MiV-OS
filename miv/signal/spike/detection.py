__doc__ = """

Code Example::

    detection = ThresholdCutoff(cutoff=3.5)
    spiketrains = detection(signal, timestamps, sampling_rate)

.. currentmodule:: miv.signal.spike

.. autosummary::
   :nosignatures:
   :toctree: _toctree/DetectionAPI

   SpikeDetectionProtocol
   ThresholdCutoff
   ExtractWaveforms
   WaveformAverage
   WaveformStatisticalFilter

"""
__all__ = [
    "ThresholdCutoff",
    "query_firing_rate_between",
    "plot_firing_rate_for_each_channel",
]


import csv
import inspect
import os
import pathlib
import time
from dataclasses import dataclass

import matplotlib.pyplot as plt
import neo
import numpy as np
from tqdm import tqdm

from miv.core.datatype import Signal, Spikestamps
from miv.core.operator.operator import OperatorMixin
from miv.core.operator.wrapper import cache_call
from miv.statistics.spiketrain_statistics import firing_rates
from miv.typing import SignalType, SpikestampsType
from miv.visualization.event import plot_spiketrain_raster


@dataclass
class ThresholdCutoff(OperatorMixin):
    """ThresholdCutoff
    Spike sorting step by step guide is well documented `here <http://www.scholarpedia.org/article/Spike_sorting>`_.

    Attributes
    ----------
        dead_time : float
            (default=0.003)
        search_range : float
            (default=0.002)
        cutoff : float | np.ndarray
            (default=5.0)
        tag : str
        units : str | pq.UnitTime
            (default='sec')
        progress_bar : bool
            Toggle progress bar (default=True)
        return_neotype : bool
            If true, return spiketrains in neo.Spiketrains (default=True)
            If false, return list of numpy-type spiketrains.
        plot_interval : float
            Interval for plotting raster, in sec.(default=10.0)
    """

    dead_time: float = 0.003
    search_range: float = 0.002
    cutoff: float = 5.0
    tag: str = "spike detection"
    progress_bar: bool = False
    units: str = "sec"
    return_neotype: bool = False  # TODO: Remove, shift to spikestamps datatype

    exclude_channels: tuple[int] = None

    num_proc: int = 1

    plot_interval: float = 10.0
    num_save: int | None = None

    def __post_init__(self):
        super().__init__()
        if self.exclude_channels is None:
            self.exclude_channels = []

    @cache_call
    def __call__(self, signal: SignalType) -> SpikestampsType:
        """Execute threshold-cutoff method and return spike stamps

        Parameters
        ----------
        signal : Signal
        custom_spike_threshold : np.ndarray
            If not None, use this value * cutoff as spike threshold.

        Returns
        -------
        spiketrain_list : list[SpikestampsType]

        """
        if not inspect.isgenerator(
            signal
        ):  # TODO: Refactor in multiprocessing-enabling decorator
            return self._detection(signal)
        else:
            collapsed_result = Spikestamps()
            # with multiprocessing.Pool(self.num_proc) as pool:
            #    #for result in pool.map(functools.partial(ThresholdCutoff._detection, self=self), signal):
            #    inputs = list(signal)
            #    print(inputs)
            #    for result in pool.map(self._detection, inputs): # TODO: Something is not correct here. Check memory usage.
            #        collapsed_result.extend(spiketrain)
            self.logger.info(f"Spike detection with {self.num_proc} processes...")
            for idx, sig in enumerate(signal):  # TODO: mp
                stime = time.time()
                collapsed_result.extend(self._detection(sig))
                self.logger.info(
                    f"Processing segment {idx}: {time.time() - stime:.02f} sec"
                )
            return collapsed_result

    # @staticmethod
    def _detection(self, signal: SignalType):
        # Spike detection for each channel
        spiketrain_list = []
        num_channels = signal.number_of_channels  # type: ignore
        timestamps = signal.timestamps
        rate = signal.rate
        for channel in tqdm(
            range(num_channels), disable=not self.progress_bar, desc=self.tag
        ):
            if channel in self.exclude_channels:
                spiketrain_list.append(np.array([]))
                continue
            array = signal[channel]  # type: ignore

            # Spike Detection: get spikestamp
            spike_threshold = self._compute_spike_threshold(array, cutoff=self.cutoff)
            crossings = self._detect_threshold_crossings(
                array, rate, spike_threshold, self.dead_time
            )
            spikes = self._align_to_minimum(array, rate, crossings, self.search_range)
            spikestamp = spikes / rate + timestamps.min()
            # Convert spikestamp to neo.SpikeTrain (for plotting)
            if self.return_neotype:
                spiketrain = neo.SpikeTrain(
                    spikestamp,
                    units=self.units,  # TODO: make this compatible to other units
                    t_stop=timestamps.max(),
                    t_start=timestamps.min(),
                )
                spiketrain_list.append(spiketrain)
            else:
                spiketrain_list.append(spikestamp.astype(np.float64))
        spikestamps = Spikestamps(spiketrain_list)
        return spikestamps

    def _compute_spike_threshold(
        self, signal: SignalType, cutoff: float = 5.0
    ) -> (
        float
    ):  # TODO: make this function compatible to array of cutoffs (for each channel)
        """
        Returns the threshold for the spike detection given an array of signal.

        Denoho D. et al., `link <https://web.stanford.edu/dept/statistics/cgi-bin/donoho/wp-content/uploads/2018/08/denoiserelease3.pdf>`_.
        Spike sorting step by step, `step 2 <http://www.scholarpedia.org/article/Spike_sorting>`_.

        Parameters
        ----------
        signal : np.array
            The signal as a 1-dimensional numpy array
        cutoff : float
            The spike-cutoff multiplier. (default=5.0)
        """
        noise_mid = np.median(np.absolute(signal)) / 0.6745
        spike_threshold = -cutoff * noise_mid
        return spike_threshold

    def _detect_threshold_crossings(
        self, signal: SignalType, fs: float, threshold: float, dead_time: float
    ):
        """
        Detect threshold crossings in a signal with dead time and return them as an array

        The signal transitions from a sample above the threshold to a sample below the threshold for a detection and
        the last detection has to be more than dead_time apart from the current one.

        Parameters
        ----------
        signal : SignalType
            The signal as a 1-dimensional numpy array
        fs : float
            The sampling frequency in Hz
        threshold : float
            The threshold for the signal
        dead_time : float
            The dead time in seconds.
        """
        dead_time_idx = dead_time * fs
        threshold_crossings = np.diff((signal <= threshold).astype(int) > 0).nonzero()[
            0
        ]
        distance_sufficient = np.insert(
            np.diff(threshold_crossings) >= dead_time_idx, 0, True
        )
        while not np.all(distance_sufficient):
            # repeatedly remove all threshold crossings that violate the dead_time
            threshold_crossings = threshold_crossings[distance_sufficient]
            distance_sufficient = np.insert(
                np.diff(threshold_crossings) >= dead_time_idx, 0, True
            )
        return threshold_crossings

    def _get_next_minimum(self, signal, index, max_samples_to_search):
        """
        Returns the index of the next minimum in the signal after an index

        :param signal: The signal as a 1-dimensional numpy array
        :param index: The scalar index
        :param max_samples_to_search: The number of samples to search for a minimum after the index
        """
        search_end_idx = min(index + max_samples_to_search, signal.shape[0])
        min_idx = np.argmin(signal[index:search_end_idx])
        return index + min_idx

    def _align_to_minimum(self, signal, fs, threshold_crossings, search_range):
        """
        Returns the index of the next negative spike peak for all threshold crossings

        :param signal: The signal as a 1-dimensional numpy array
        :param fs: The sampling frequency in Hz
        :param threshold_crossings: The array of indices where the signal crossed the detection threshold
        :param search_range: The maximum duration in seconds to search for the minimum after each crossing
        """
        search_end = int(search_range * fs)
        aligned_spikes = [
            self._get_next_minimum(signal, t, search_end) for t in threshold_crossings
        ]
        return np.array(aligned_spikes)

    def plot_spiketrain(
        self,
        spikestamps,
        inputs,
        show: bool = False,
        save_path: pathlib.Path | None = None,
    ) -> plt.Axes:
        """
        Plot spike train in raster
        """
        t0 = spikestamps.get_first_spikestamp()
        tf = spikestamps.get_last_spikestamp()

        # TODO: REFACTOR. Make single plot, and change xlim
        term = self.plot_interval
        n_terms = int(np.ceil((tf - t0) / term))
        if n_terms == 0:
            # TODO: Warning message
            return None
        for idx in range(n_terms):
            if self.num_save is not None and idx > self.num_save:
                break
            fig, ax = plot_spiketrain_raster(
                spikestamps, idx * term + t0, min((idx + 1) * term + t0, tf)
            )
            if save_path is not None:
                plt.savefig(
                    os.path.join(save_path, f"spiketrain_raster_{idx:03d}.png"),
                    format="png",
                    dpi=300,
                )
            if not show:
                plt.close("all")
        if show:
            plt.show()

    def plot_firing_rate_histogram(
        self, spikestamps, inputs, show=False, save_path=None
    ):
        """Plot firing rate histogram"""
        threshold = 3

        rates = firing_rates(spikestamps)["rates"]
        hist, bins = np.histogram(rates, bins=20)
        logbins = np.logspace(
            np.log10(max(bins[0], 1e-3)), np.log10(bins[-1]), len(bins)
        )
        fig = plt.figure()
        ax = plt.gca()
        ax.hist(rates, bins=logbins)
        ax.axvline(
            np.mean(rates),
            color="r",
            linestyle="dashed",
            linewidth=1,
            label=f"Mean {np.mean(rates):.2f} Hz",
        )
        ax.axvline(
            threshold,
            color="g",
            linestyle="dashed",
            linewidth=1,
            label="Quality Threshold",
        )
        ax.set_xscale("log")
        xlim = ax.get_xlim()
        ax.set_xlabel("Firing rate (Hz) (log-scale)")
        ax.set_ylabel("Count")
        ax.set_xlim([min(xlim[0], 1e-1), max(1e2, xlim[1])])
        ax.legend()
        if save_path is not None:
            fig.savefig(os.path.join(f"{save_path}", "firing_rate_histogram.png"))
            with open(
                os.path.join(f"{save_path}", "firing_rate_histogram.csv"), "w"
            ) as f:
                writer = csv.writer(f)
                writer.writerow(["channel", "firing_rate_hz"])
                data = list(enumerate(rates))
                data.sort(reverse=True, key=lambda x: x[1])
                for ch, rate in data:
                    writer.writerow([ch, rate])
        if show:
            plt.show()

        return ax

    def plot_firing_rate_instantaneous(
        self, spikestamps, inputs, show=False, save_path=None
    ):
        """Plot firing rate throughout time"""
        binsize = 0.10
        binned_spiketrain = spikestamps.binning(binsize, return_count=True)
        time_minute = binned_spiketrain.timestamps / 60  # minute
        time_minute -= time_minute[0]  # Reset the time to zero
        mean_firing_rate = (
            binned_spiketrain.data.mean(axis=binned_spiketrain._CHANNELAXIS) / binsize
        )

        fig = plt.figure()
        plt.plot(time_minute, mean_firing_rate)
        plt.ylabel("Average (channel-wise) Firing rate (Hz)")
        plt.xlabel("Time (minute)")
        if save_path is not None:
            fig.savefig(os.path.join(save_path, "instantaneous_firing_rate.png"))
            with open(
                os.path.join(save_path, "instantaneous_firing_rate.csv"), "w"
            ) as f:
                writer = csv.writer(f)
                writer.writerow(["time", "mean_firing_rate"])
                for tm, mfr in zip(time_minute, mean_firing_rate):
                    writer.writerow([tm, mfr])
        if show:
            plt.show()
        plt.close("all")


def plot_firing_rate_for_each_channel(spikestamps, inputs, show=False, save_path=None):
    if save_path is None:
        return
    """Plot firing rate throughout time"""
    binsize = 0.10
    binned_spiketrain = spikestamps.binning(binsize, return_count=True)
    time_minute = binned_spiketrain.timestamps / 60  # minute
    time_minute -= time_minute[0]  # Reset the time to zero

    fname = "channel_firing_rates"
    save_path = os.path.join(save_path, fname)
    os.makedirs(save_path, exist_ok=True)

    for channel in range(binned_spiketrain.number_of_channels):
        firing_rate = binned_spiketrain[channel] / binsize

        fig = plt.figure()
        plt.plot(time_minute, firing_rate)
        plt.ylabel(f"Average (channel-wise) Firing rate (Hz) ch{channel:04d}")
        plt.xlabel("Time (minute)")
        fig.savefig(os.path.join(save_path, f"firing_rate_ch{channel:04d}.png"))
        plt.close("all")
        with open(os.path.join(save_path, "instantaneous_firing_rate.csv"), "w") as f:
            writer = csv.writer(f)
            writer.writerow(["time", "firing_rate"])
            for tm, fr in zip(time_minute, firing_rate):
                writer.writerow([tm, fr])


@dataclass
class ThresholdCutoffNonSparse(ThresholdCutoff):
    bin_size: float = 0.002

    def __call__(self, signal: SignalType) -> SpikestampsType:
        for idx, sig in enumerate(signal):  # TODO: mp
            spikestamps = self._detection(sig)
            binned = self._binning(sig, spikestamps)
            yield spikestamps

    def _binning(
        self,
        signal,
        spikestamps,
        t_start: float | None = None,
        t_end: float | None = None,
    ) -> Signal:
        t_start = signal.timestamps[0]
        t_end = signal.timestamps[0]

        n_bins = int(np.ceil((t_end - t_start) / self.bin_size))
        time = t_start + (np.arange(n_bins + 1) * self.bin_size)

        num_channels = spikestamps.number_of_channels
        signal = Signal(
            data=np.zeros(
                [n_bins, num_channels],
                dtype=np.int_,
            ),
            timestamps=time[:-1],
            rate=1.0 / self.bin_size,
        )
        for idx, spiketrain in enumerate(self.data):
            bins = np.digitize(spiketrain, time)
            bincount = np.bincount(bins, minlength=n_bins + 2)[1:-1]
            bin_spike = bincount
            signal.data[:, idx] = bin_spike
        return signal


def query_firing_rate_between(
    spikestamps: Spikestamps,
    min_firing_rate: float,
    max_firing_rate: float,
) -> np.ndarray:
    """
    Mask channels with firing rates between min_firing_rate and max_firing_rate

    Parameters
    ----------
    spikestamps : Spikestamps
        Spikestamps
    min_firing_rate : float
        Minimum firing rate
    max_firing_rate : float
        Maximum firing rate

    Returns
    -------
    Spikestamps
        Mask of channels with firing rates between min_firing_rate and max_firing_rate
    """
    rates = np.array(firing_rates(spikestamps)["rates"])
    masks = np.logical_and(rates >= min_firing_rate, rates <= max_firing_rate)
    return np.where(masks)[0]
