# MPI Support for Data Loading

This document describes the MPI (Message Passing Interface) support parallel data loading module.
Feature remains experimental.

## Requirements

To use MPI functionality and `mpi4py`.

## Usage

### Basic MPI Usage

```python
# from miv.io.openephys import Data
from miv.io.intan import DataIntan as Data
from miv.core.pipeline import Pipeline
from miv.signal.filter import ButterBandpass
from miv.signal.spike import ThresholdCutoff

# Get MPI communicator
comm = MPI.COMM_WORLD

# Create DataIntan instance
data = DataIntan("/path/to/intan/data")
data.configure_load(mpi_comm=comm)  # Key to include comm in the load

bandpass_filter = ButterBandpass(lowcut=300, highcut=3000, order=4)
spike_detection = ThresholdCutoff(cutoff=4.0)
spike_detection.runner = SupportMPIMerge(comm=comm)

data >> bandpass_filter >> spike_detection
pipeline = Pipeline(spike_detection)
pipeline.run()
```

### Running with MPI

```bash
mpirun -np 4 python your_script.py
```
