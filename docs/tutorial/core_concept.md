Concepts
========

Core Idea
---------

The `MiV-OS` is a community-base project that aims to provide a set of tools for the analysis of the electropysiology recordings. We are always welcome to new contributors, and we are always looking for new ideas.

The amount of data that can be collected from electrophysiology experiments is increasing exponentially, while the amount of time that a researcher can spend on the analysis of the data is limited. Unfortunately, this limitation is often due to the lack of tools that can be adapted and automated for an individual experiments. We aims to bridge this gap by providing a set of pipelining tools and templates that can be customized and extended for any purposed experiments.

Key features and developing philosophy evolves around followings concepts:

1. Easy pipeline construction and visualization
2. Automatic caching the data processing to avoid re-computation of the expensive operations
3. Lazy operation run to adapt different parallelization paradigms, for both multiprocessing and MPI.
4. Modular and customizable design to adapt varying data format and experimental protocols.

Key Modules
-----------

The `MiV-OS` is a collection of modules that can be used independently. The key modules includes:

- Core
    - core.datatype: extension from native `python` data-type or `numpy` data-type with additional functionality.
        - ex) `miv.core.datatype.Timestamps`, `miv.core.datatype.Signal`
    - core.operator: functional protocols that takes parameters and some `core.datatype` and yield another set of `core.datatype`.
    - core.pipeline: a pipeline that can be constructed from `core.operator` and `core.datatype`.
    - core.policy: a set of policies that can be used to control the behavior of the `core.pipeline`.
- DataLoader
    - Serves as a `source` or the `starting` node of the analysis pipeline.
    - Designed to accomodate the difference in the data format.
- Operator
    - A set of operation that processes the data.
    - Each `operator` can be treated as a functional node in the pipeline.
    - Comply `Callback`: can be used to monitor the progress or insert custom codes. ex) plotting.
    - Comply `Cachable`: automatically cache the results. Caching policy can be controlled.
    - Comply `Chainable`: can be chained with other `operator` to form a new `operator`.
    - Comply `Runnable`: can be called and run independently or within a given pipeline.
