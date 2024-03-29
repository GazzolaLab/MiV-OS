***********************************************************
Causality/Connectivity (:mod:`miv.statistics.connectivity`)
***********************************************************

Connectivity Operators
======================

.. currentmodule:: miv.statistics.connectivity

.. autosummary::
   :nosignatures:
   :toctree: _toctree/ConnectivityAPI

   DirectedConnectivity
   plot_eigenvector_centrality


Useful External Packages
========================

Here are few external `python` packages that can be used for further causality analysis.

scipy signal
------------

- `scipy coherence analysis <https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.coherence.html>`_

.. autosummary::

   scipy.signal.coherence
   scipy.signal.welch
   scipy.signal.csd

elephant connectivity
---------------------

- `elephant documentation <https://elephant.readthedocs.io/>`_

.. autosummary::

   elephant.causality.granger.pairwise_granger
