Installation
============

This page explains how to install `MiV-OS` on your system.
If you are familiar with Python and have a Python environment set up, you can install `MiV-OS` using `pip`.

:code:`pip install MiV-OS`

You can also download the source code from `GitHub <https://github.com/GazzolaLab/MiV-OS>`_ directly to install
latest version of `MiV-OS`.

.. code-block:: bash

    cd working-directory
    git clone https://github.com/GazzolaLab/MiV-OS.git
    cd MiV-OS
    pip install .

Conda installation
------------------

We highly recommend using `conda <https://docs.conda.io/en/latest/>`_ to manage your Python environments, since
`MiV-OS` utilize other scientific packages that may conflict with your existing Python environment.

.. code-block:: bash

    conda create --name miv python=3.10.3
    conda activate myenv
    pip install MiV-OS

Poetry installation
-------------------

If you are planning to develop `MiV-OS`, we recommend using `Poetry <https://python-poetry.org/>`_ to manage your Python environments and dependencies.
Install Poetry (if not already installed) using the command appropriate for your system from the official Poetry installation documentation.

.. code-block:: bash

    cd working-directory
    git clone https://github.com/GazzolaLab/MiV-OS.git
    cd MiV-OS
    make install

Requirements
------------

Before installing `MiV-OS`, please ensure you have the following installed:

- Python 3 (3.8+ are supported)

Following packages are optionally required for some features:

- FFmpeg (optional for video generation)
- MPI (optional for parallelization)
- Poetry (optional for development)


Troubleshooting
---------------

If you have any issues during the installation please post a `GitHub-issue <https://github.com/GazzolaLab/MiV-OS/issues>`_ with the details.
