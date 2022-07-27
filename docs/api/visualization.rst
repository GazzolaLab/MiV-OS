***************************************************
Visualization Tools (:mod:`miv.visualization`)
***************************************************

Plotting Tools
==============

DFT Plot
--------

.. currentmodule:: miv.visualization

.. automodule:: miv.visualization.fft_domain

   .. autosummary::
      :nosignatures:
      :toctree: _toctree/VisualizationAPI

      plot_frequency_domain
      plot_spectral

Spike Waveform Overlap
----------------------

.. currentmodule:: miv.visualization

.. automodule:: miv.visualization.waveform

   .. autosummary::
      :nosignatures:
      :toctree: _toctree/VisualizationAPI

      extract_waveforms
      plot_waveforms

Causality Analysis
------------------

.. currentmodule:: miv.visualization

.. automodule:: miv.visualization.causality

   .. autosummary::
      :nosignatures:
      :toctree: _toctree/VisualizationAPI

      pairwise_causality_plot
      spike_triggered_average_plot

Burst Analysis
------------------

.. currentmodule:: miv.visualization

.. automodule:: miv.visualization.event

   .. autosummary::
      :nosignatures:
      :toctree: _toctree/VisualizationAPI

      plot_burst

Connectivity Plots
------------------

.. currentmodule:: miv.visualization

.. automodule:: miv.visualization.plot_connectivity

   .. autosummary::
      :nosignatures:
      :toctree: _toctree/VisualizationAPI

      plot_connectivity
      plot_connectivity_interactive


Useful External Packages
========================

Here are few external `python` packages that can also be used for visualization.

Viziphant
---------

`viziphant (elephant) documentation: <https://viziphant.readthedocs.io/en/latest/modules.html>`_

.. autosummary::

   viziphant.rasterplot.rasterplot
   viziphant.rasterplot.rasterplot_rates
   viziphant.spike_train_correlation.plot_corrcoef
