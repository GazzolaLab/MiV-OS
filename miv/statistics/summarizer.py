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

from typing import Any, Dict, Iterable, Optional, Union

import datetime

import elephant.statistics
import matplotlib.pyplot as plt
import neo
import numpy as np
import quantities as pq
import scipy
import scipy.signal


# FIXME: For now, we provide the free function for simple usage. For more
# advanced statistical analysis, we should have a module wrapper.
def spikestamps_statistics(
    spiketrains: Union[pq.Quantity, Iterable[neo.core.SpikeTrain]],
    # t_start: Optional[float] = None,
    # t_stop: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Process basic spiketrains statistics: rates, mean, variance.

    Parameters
    ----------
    spiketrains : Iterable[neo.core.SpikeTrain]

    Returns
    -------
    Iterable[Any]

    """
    rates = [
        float(elephant.statistics.mean_firing_rate(spikestamp).magnitude)
        for spikestamp in spiketrains
    ]
    rates_mean_over_channel = np.mean(rates)
    rates_variance_over_channel = np.var(rates)
    return {
        "rates": rates,
        "mean": rates_mean_over_channel,
        "variance": rates_variance_over_channel,
    }
