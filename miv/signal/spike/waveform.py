__doc__ = """
Module for extracting each spike waveform and visualize.
"""
__all__ = ["ExtractWaveforms", "WaveformAverage"]

from typing import Any, Dict, Generator, List, Optional, Tuple, Union

import inspect
import os
import pathlib
from dataclasses import dataclass

import matplotlib
import matplotlib.pyplot as plt
import neo
import numpy as np
import quantities as pq
from scipy.signal import lfilter, savgol_filter
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from miv.core.datatype import Signal, Spikestamps
from miv.core.operator import OperatorMixin
from miv.core.wrapper import wrap_generator_to_generator
from miv.mea import MEAGeometryProtocol
from miv.typing import SignalType, SpikestampsType
from miv.core.wrapper import wrap_cacher


@dataclass
class ExtractWaveforms(OperatorMixin):
    """
    Extract spike waveforms as signal cutouts around each spike index as a spikes x samples numpy array

    Parameters
    ----------
    channels : Optional[List[int]]
        Interested channels. If None, extract from all channels. (default: None)
    pre : pq.Quantity | float
        The duration of the cutout before the spike in seconds. (default: 0.001 s)
    post : pq.Quantity | float
        The duration of the cutout after the spike in seconds. (default: 0.002 s)
    plot_n_spikes : Optional[int]
        The number of cutouts to plot. None to plot all. (default: 100)
    """

    channels: Optional[List[int]] = None
    pre: pq.Quantity = 0.001 * pq.s
    post: pq.Quantity = 0.002 * pq.s
    plot_n_spikes: int = 100
    tag: str = "extract waveform"

    progress_bar: bool = False

    def __post_init__(self):
        super().__init__()

    @wrap_cacher("waveform")
    def __call__(
        self, signal: Generator[Signal, None, None], spikestamps: Spikestamps
    ) -> Dict[int, Signal]:
        """__call__

        Parameters
        ----------
        signal : Generator[Signal, None, None]
            The signal to extract the event feature
        spikestamps : Spikestamps
            The sample index of all spikes as a 1-dim numpy array

        Returns
        -------
        Stack of spike cutout: List[np.ndarray]
            Return stacks of spike cutout; shape(n_spikes, width).

        """

        if isinstance(self.pre, pq.Quantity):
            pre = self.pre.rescale(pq.s).magnitude
        else:
            pre = self.pre
        if isinstance(self.post, pq.Quantity):
            post = self.post.rescale(pq.s).magnitude
        else:
            post = self.post
        if not inspect.isgenerator(signal):
            signal = [signal]

        waveforms = {}
        previous_sig = None
        for sig in tqdm(signal, desc="For each Signal segments"):
            sampling_rate = sig.rate
            num_channels = sig.number_of_channels
            channels = range(num_channels) if self.channels is None else self.channels
            pre_idx = int(pre * sampling_rate)
            post_idx = int(post * sampling_rate)
            assert (
                pre_idx + post_idx > 0
            ), "Set larger pre/post duration. pre+post duration must be more than 1/sampling_rate."
            spikestamps_view = spikestamps.get_view(
                sig.get_start_time(), sig.get_end_time()
            )
            for ch in channels:
                # Padding signal
                spikestamp = spikestamps_view[ch]
                if len(spikestamp) == 0:
                    continue
                padded_signal = np.pad(
                    sig.data[:, ch], ((pre_idx, post_idx),), constant_values=0
                )
                if previous_sig is not None:
                    padded_signal[:pre_idx] = previous_sig.data[-pre_idx:, ch]
                cutout = np.empty(
                    [pre_idx + post_idx, len(spikestamp)], dtype=np.float_
                )
                for idx, time in enumerate(spikestamp):
                    index = int(round((time - sig.get_start_time()) * sampling_rate))
                    if index + post_idx + pre_idx >= sig.data.shape[0]:
                        # FIXME: need better algorithm for handling segmented signal
                        continue
                    cutout[:, idx] = padded_signal[index : (index + post_idx + pre_idx)]
                if ch not in waveforms:
                    waveforms[ch] = Signal(
                        data=cutout,
                        timestamps=np.arange(pre_idx + post_idx).astype(np.float_)
                        / sampling_rate
                        - pre,
                        rate=sampling_rate,
                    )
                else:
                    waveforms[ch].append(cutout)
            previous_sig = sig

        return waveforms

    def plot_waveforms(
        self,
        waveforms: Dict[int, Signal],
        show: bool = False,
        save_path: Optional[pathlib.Path] = None,
        plot_kwargs: Dict[Any, Any] = None,
    ):
        """
        Plot an overlay of spike waveforms

        Parameters
        ----------
        waveforms : np.ndarray
            A spikes x samples array of cutouts waveforms.
        plot_kwargs : Dict[Any, Any]
            Addtional keyword-arguments for matplotlib.pyplot.plot.
        """
        for ch, signal in waveforms.items():
            cutout = signal.data
            num_cutout = signal.number_of_channels
            if self.plot_n_spikes is None:
                self.plot_n_spikes = num_cutout
            plot_n_spikes = min(self.plot_n_spikes, num_cutout)
            if num_cutout == 0:
                continue

            if not plot_kwargs:
                plot_kwargs = {"alpha": 0.3, "linewidth": 1, "color": "b"}

            time = signal.timestamps * pq.s
            time = time.rescale(pq.ms).magnitude

            fig, ax = plt.subplots(figsize=(8, 5))
            for i in range(plot_n_spikes):
                arr = cutout[:, i]
                arr[np.isnan(arr)] = 0
                ax.plot(
                    time,
                    arr,
                    **plot_kwargs,
                )
            ax.set_xlabel("Time (ms)")
            ax.set_ylabel("Voltage (microV)")
            ax.set_title(f"Spike Cutouts (channel {ch})")
            if save_path:
                plt.savefig(os.path.join(save_path, f"spike_cutouts_ch{ch:03}.png"))
            plt.close(fig)
        plt.close("all")


@dataclass
class WaveformAverage(OperatorMixin):
    """
    Plot the average waveform of each channel
    """

    tag: str = "waveform average"

    def __post_init__(self):
        super().__init__()

    def __call__(self, waveforms: Dict[int, Signal], mea: MEAGeometryProtocol):
        nrow, ncol = mea.nrow, mea.ncol

        fig, axes = plt.subplots(nrow, ncol, figsize=(ncol * 6, nrow * 4))
        for key, cutout in waveforms.items():
            idx = mea.get_ixiy(key)
            if idx is None:
                continue
            axes[idx[0], idx[1]].plot(
                cutout.timestamps, cutout.data.mean(axis=cutout._CHANNELAXIS)
            )
        # Bottom row
        for i in range(ncol):
            axes[-1, i].set_xlabel("time (s)")

        # Left row
        for i in range(nrow):
            axes[i, 0].set_ylabel("Voltage (microV)")

        plt.suptitle("Waveform Average Plot")

        os.makedirs(self.analysis_path, exist_ok=True)
        plt.savefig(os.path.join(self.analysis_path, "waveform_average.png"))
        plt.close(fig)
