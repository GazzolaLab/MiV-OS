__all__ = [
    "SignalPlot"
]


import csv
import inspect
import os
import pathlib
import time
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np

from miv.core.datatype import Signal, Spikestamps
from miv.core.operator.operator import OperatorMixin
from miv.core.operator.wrapper import cache_call
from miv.typing import SignalType, SpikestampsType
from miv.signal.utils import downsample_average


@dataclass
class SignalPlot(OperatorMixin):
    """
    Current imple copy memory state. Should be fixed in the future.
    """

    tag: str = "spike detection"
    plot_interval: float = 10.0
    num_save: int | None = None

    def __post_init__(self):
        super().__init__()

    # @cache_call  # Plotting only. No cahce needed
    def __call__(self, signal: SignalType) -> SignalType:
        if not inspect.isgenerator(signal):
            raise NotImplementedError
        result = None
        # TODO: Save memory stamp
        for idx, sig in enumerate(signal):  # TODO: mp
            if result is None:
                result = sig
            else:
                result.extend(sig)
        return result

    def plot_spiketrain(
        self,
        outputs,
        inputs,
        show: bool = False,
        save_path: pathlib.Path | None = None,
    ) -> plt.Axes:
        """
        Plot spike train in raster
        """
        signal = outputs.data
        timestamps = outputs.timestamps

        tf = timestamps[-1]
        t0 = timestamps[0]
        term = self.plot_interval
        n_terms = int(np.ceil((tf - t0) / term))

        for channel in range(outputs.number_of_channels):
            signal = outputs[channel]
            _save_path = os.path.join(save_path, str(channel))
            os.makedirs(_save_path, exist_ok=True)

            if n_terms == 0:
                continue

            for idx in range(n_terms):
                start_time = idx * term + t0
                end_time = (idx+1) * term + t0
                
                # Find start and end indices using searchsorted (more efficient for sorted arrays)
                start_idx = np.searchsorted(timestamps, start_time)
                end_idx = np.searchsorted(timestamps, end_time)
                
                # Get the segment data
                segment_timestamps = timestamps[start_idx:end_idx]
                segment_signal = signal[start_idx:end_idx]
                
                # Downsample to 128 points
                n_points = 128
                downsampled_timestamps, downsampled_signal = downsample_average(
                    segment_timestamps, segment_signal, n_points
                )
                
                # Create new figure for each segment
                fig, ax = plt.subplots(1, 1, figsize=(8, 6))
                ax.plot(downsampled_timestamps, downsampled_signal)
                ax.set_xlabel("Time (sec)")
                ax.set_title(f"Signal ch: {channel}, {start_time:.03f} - {end_time:.03f} sec")
                ax.set_xlim(timestamps[start_idx], timestamps[start_idx] + term)
                
                if save_path is not None:
                    plt.savefig(
                        os.path.join(_save_path, f"signal_{idx:03d}.png"),
                        dpi=300,
                    )
                plt.close("all")
