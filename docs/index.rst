MiV-OS
======

Data analysis often begins as a straight line: load something, transform it,
inspect the result. As an investigation grows, that line becomes a network of
shared steps, alternative measurements, expensive computations, and results
that must still be understandable months later.

**MiV-OS is a Python framework for making that network explicit.** An analysis
is assembled from small operators connected as a graph. The graph describes
the scientific path from data to result; caching, streaming, callbacks, and
runners control how that path is carried out.

MiV-OS grew from the `Mind in Vitro <https://mindinvitro.illinois.edu>`_
project and has a rich collection of electrophysiology tools. You do not need
to work in electrophysiology to use its central idea: analysis should remain
composable, inspectable, and adaptable as a question or dataset changes.

Begin in :doc:`about` for the philosophy and a map of the documentation.
If you already know what you need, choose a room below and enter directly.


.. toctree::
   :maxdepth: 2
   :caption: Overview

   about
   tutorial/index
   guide/index
   api/index
   discussion/index

.. toctree::
   :maxdepth: 1
   :caption: Architecture

   adr/0001-streaming-dag-execution-and-cache-replay

Contributing
------------

MiV-OS is free and open source, developed and maintained by the Gazzola Lab at
the University of Illinois Urbana-Champaign. Questions, corrections, examples,
and code contributions are welcome; see the `contribution guide
<https://github.com/GazzolaLab/MiV-OS/blob/main/CONTRIBUTING.md>`_ to begin.

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
