# Concepts

## Core Idea

The `MiV-OS` is a community-base project that aims to provide a set of tools for the analysis of the electropysiology recordings. We are always welcome to new contributors, and we are always looking for new ideas.

The amount of data that can be collected from electrophysiology experiments is increasing exponentially, while the amount of time that a researcher can spend on the analysis of the data is limited. Unfortunately, this limitation is often due to the lack of tools that can be adapted and automated for an individual experiments. We aims to bridge this gap by providing a set of pipelining tools and templates that can be customized and extended for any purposed experiments.

Key features and developing philosophy evolves around followings concepts:

1. Easy pipeline construction and visualization
2. Automatic caching the data processing to avoid re-computation of the expensive operations
3. Lazy operation run to adapt different parallelization paradigms, for both multiprocessing and MPI.
4. Modular and customizable design to adapt varying data format and experimental protocols.

### Chaining and Graph-based Data Flow

The pipeline system in `MiV-OS` aims lazy execution and loading: each operators are constructed as a standalone object, and heavy operations are only executed when output is requested explicitly.
Each nodes are isolated, and they can be chained to form a graph of data processing. Some nodes, like IO or Data, only makes output (downstream) connections, using `>>` operator. They are called `source nodes`.
Some node may only have input (upstream) connections, and they are called `end nodes`, such as plotter or visualizer nodes that does not produce any output.
Generally, a node can form both upstream and downstream connections, where operator are in charge of processing the data and producing the output.

```mermaid
graph LR
    Data("Data Node")
    Operator("Operator Node")
    End("End Node")
    Data -- ">>" --> Operator
    Operator -- ">>" --> End
```

This diagram illustrates how a data node (such as `Spikestamps` or `IONode`) can be chained to an operator node using the `>>` operator in MiV-OS pipelines.

## Core Module

The `core` module provides the foundational tools for building data processing pipelines. To build custom nodes, an operator mixin is provided for each type.

### Datatype (Source)

Extended data types that inherit from native Python or NumPy types with additional functionality for electrophysiology data.

- **Signal**: A 2D array representing continuous signal data with timestamps
  - Data shape: `[signal_length, number_of_channels]`
  - Includes timestamps and sampling rate

- **Spikestamps**: A list of arrays representing spike times for each channel
  - Each channel contains an array of spike timestamps
  - Supports channel-wise operations and binning

- **Events**: A list of event timestamps
  - Represents discrete events in time
  - Can be binned into Signal format

All datatypes inherit from `DataNodeMixin`, which provides chaining capabilities and makes them compatible with the pipeline system.

### OperatorMixin

The `OperatorMixin` is a mixin to create general purpose `node` that can be used in a pipeline. Operators are composed of several features that provide multiple capabilities.

**Chaining**: Operators can be chained using the `>>` operator:

```python
operator1 >> operator2 >> operator3
```

This creates a dependency graph where `operator1` is upstream of `operator2`, etc.

**Caching**: Automatic result caching with configurable policies.

- `ON`: Use cache if available, otherwise compute and save (default)
- `OFF`: Never use cache, always compute
- `MUST`: Must use cache, raise error if not available
- `OVERWRITE`: Always compute and overwrite existing cache

**Callbacks**: Support for custom callbacks:

- `after_run_*`: Methods called after operator execution
- `plot_*`: Methods called for visualization (can be skipped with `skip_plot=True`)
- Callbacks can be dynamically added using the `<<` operator

**Runner Policies**: Different execution strategies for parallelization:

- `VanillaRunner`: Default sequential execution (supports MPI broadcast if available)
- `StrictMPIRunner`: MPI-based execution where each rank processes independently
- `SupportMPIMerge`: MPI execution with automatic result merging

### OperatorGeneratorMixin

Specialized operators for generator-based processing. Most of the features are similar to `OperatorMixin`, but with additional features for generator-based processing.
Generator operators are useful for processing data streams or when operations naturally yield multiple results.

**Runner Policies**: Different execution strategies for parallelization:

- `VanillaGeneratorRunner`: Default sequential execution (supports MPI broadcast if available)
- `MultiprocessingGeneratorRunner`: Parallel execution using Python multiprocessing

## Pipeline

The `Pipeline` class orchestrates the execution of operator graphs:

- Automatic Topological Sorting: Determines execution order based on operator dependencies
- Cache-Aware Execution: Skips operators with valid cached results
- Flexible Execution: Can run single nodes or multiple independent nodes
- Directory Management: Supports separate working, cache, and temporary directories
- Visualization: Can summarize execution order for debugging

The pipeline automatically resets callbacks for all nodes, sets save paths for all nodes, executes nodes in dependency order (topological sort), and handles errors and provides verbose logging.
