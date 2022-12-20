******************************************
Statistics Modules (:mod:`miv.statistics`)
******************************************

Statistics Tools
================

.. currentmodule:: miv.statistics

Signals
-------

.. autosummary::
  :nosignatures:
  :toctree: _toctree/StatisticsAPI

  signal_to_noise
  spike_amplitude_to_background_noise

Spikestamps Statistics
----------------------

.. autosummary::
  :nosignatures:
  :toctree: _toctree/StatisticsAPI

  firing_rates
  interspike_intervals
  coefficient_variation
  binned_spiketrain
  fano_factor

Burst Analysis
------------------

.. autosummary::
  :nosignatures:
  :toctree: _toctree/StatisticsAPI

  burst

Information Theory
------------------

.. autosummary::
  :nosignatures:
  :toctree: _toctree/StatisticsAPI

  shannon_entropy
  block_entropy
  entropy_rate
  active_information
  mutual_information
  relative_entropy
  conditional_entropy
  transfer_entropy


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
