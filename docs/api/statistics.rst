******************
Statistics Modules
******************

Statistics Tools
================

.. currentmodule:: miv.statistics

Signals
-------

.. autosummary::
  :nosignatures:
  :toctree: _toctree/StatisticsAPI

  signal_to_noise

Spikestamps Statistics
----------------------

.. autosummary::
  :nosignatures:
  :toctree: _toctree/StatisticsAPI

  firing_rates
  interspike_intervals
  coefficient_variation

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
