__doc__ = """

Statistics Tools
================

Spikestamps
-----------

.. currentmodule:: miv.statistics

.. autosummary::
  :nosignatures:
  :toctree: _toctree/StatisticsAPI

  spikestamps_statistics

Useful External Packages
========================

Here are few external `python` packages that can be used for further statistical analysis.

scipy statistics
----------------

`scipy <https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.describe.html>`_

.. autosummary::

   scipy.stats.describe

elephant.statistics
-------------------

`elephant documentation: <https://elephant.readthedocs.io/en/latest/reference/statistics.html>`_

.. autosummary::

   elephant.statistics.mean_firing_rate
   elephant.statistics.instantaneous_rate
"""
__all__ = ["spikestamps_statistics"]

from typing import Any, Optional, Iterable, Union, Dict
import numpy as np
import matplotlib.pyplot as plt
import datetime
import scipy
import scipy.signal

import neo
import elephant.statistics


# FIXME: For now, we provide the free function for simple usage. For more
# advanced statistical analysis, we should have a module wrapper.
def spikestamps_statistics(
    spikestamps: Union[np.ndarray, Iterable[float], Iterable[neo.core.SpikeTrain]],
    t_start: Optional[float] = None,
    t_stop: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Process basic spikestamps statistics: rates, mean, variance.

    Parameters
    ----------
    spikestamps : Iterable[neo.core.SpikeTrain]
    t_start : Optional[float]
    t_stop : Optional[float]

    Returns
    -------
    Iterable[Any]

    """
    rates = [
        elephant.statistics.mean_firing_rate(spikestamp, t_start, t_stop, axis=0)
        for spikestamp in spikestamps
    ]
    rates_mean_over_channel = np.mean(rates)
    rates_variance_over_channel = np.var(rates)
    return {
        "rates": rates,
        "mean": rates_mean_over_channel,
        "variance": rates_variance_over_channel,
    }
