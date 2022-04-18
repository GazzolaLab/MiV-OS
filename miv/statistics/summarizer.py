__doc__ = ""
__all__ = ["StatisticsSummary"]

from typing import Any, Optional, Iterable, Union
import numpy as np
import matplotlib.pyplot as plt
import datetime
import scipy
import scipy.signal

import neo
import elephant.statistics


class StatisticsSummary:
    """StatisticsSummary."""

    def __init__(self):
        pass

    def spikestamps_summary(
        self,
        spikestamps: Iterable[neo.core.SpikeTrain],
        t_start: Optional[float] = None,
        t_stop: Optional[float] = None,
    ) -> Iterable[Any]:
        rates = elephant.statistics.mean_firing_rate(
            spikestamps, t_start, t_stop, axis=0
        )
        rates_mean_over_channel = np.mean(rates)
        rates_variance_over_channel = np.var(rates)
        return {
            "rates": rates,
            "mean": rates_mean_over_channel,
            "variance": rates_variance_over_channel,
        }
