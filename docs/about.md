# About

**MiV-OS turns an analysis from a sequence of instructions into a visible
graph of scientific steps.**

That distinction matters when an analysis stops being a straight line. A
shared preprocessing step may feed several measurements; a result may be
expensive to reproduce; the same method may need to run on one recording today
and a collection tomorrow. MiV-OS keeps those relationships in the analysis
itself instead of hiding them in control flow and file conventions.

```{mermaid}
flowchart LR
    recording[Recording] --> prepare[Prepare signal]
    prepare --> rate[Firing rate]
    prepare --> network[Connectivity]
    prepare --> checks[Quality checks]
```

This is not a diagram drawn after the code. It is the program's structure. The
shared path is represented once, and every outcome retains a route back to its
source.

## What the graph changes

| Mechanism | Practical consequence |
| --- | --- |
| Operators represent one source or transformation | Scientific steps can be tested, replaced, and reused independently. |
| Edges represent dependencies | Shared work and branches are visible rather than implied by statement order. |
| A pipeline starts from requested results | MiV-OS discovers the upstream work needed to produce them. |
| Cache policies, streaming operators, callbacks, and runners are separate choices | Execution and diagnostics can change without rewriting the scientific graph. |

MiV-OS does not make every analysis scale automatically. Data layout, operator
design, and computing resources still matter. It gives those decisions clear
places to live so the analysis can evolve without becoming one inseparable
script.

## Built in electrophysiology, useful beyond it

MiV-OS grew from the [Mind in Vitro](https://mindinvitro.illinois.edu)
project, where long electrophysiology recordings move through repeated
preprocessing, detection, measurement, and visualization. The package includes
loaders, signal processing, spike detection and sorting, statistics, and
visualization tools for that work.

Electrophysiology is the package's first and richest collection, not a
requirement of its pipeline model. Other experimental or sequential data can
use custom loaders and operators while retaining the same graph, cache,
callback, and runner machinery.

## Is it the right level of structure?

| MiV-OS is useful when… | A plain function or notebook may be clearer when… |
| --- | --- |
| transformations are reused across analyses or datasets | the calculation is inexpensive and genuinely one-off |
| several results share preprocessing | the analysis has only a few linear steps |
| intermediate work is costly or worth inspecting | there is no value in retaining intermediate state |
| scientific logic should be separated from execution and diagnostics | introducing a graph would add more structure than understanding |

MiV-OS is an analysis framework, not a general-purpose distributed workflow
scheduler. Its value appears when the graph helps you understand, reuse, or
operate the analysis.

## Continue from here

- **Understand the model:** [Core overview](tutorial/core_concept.md)
- **Build a working graph:** [Core quickstart](tutorial/core_quickstart.md)
- **Adapt an existing analysis:** [How-to guides](guide/index.rst)
- **Browse available objects:** [API reference](api/index.rst)
- **Study execution details:** [Advanced core](tutorial/core_advanced.md)

MiV-OS is free and open source, developed and maintained by the Gazzola Lab at
the University of Illinois Urbana-Champaign. Corrections, examples, and code
contributions are welcome; see the [contribution
guide](https://github.com/GazzolaLab/MiV-OS/blob/main/CONTRIBUTING.md).

```{toctree}
:hidden:
:maxdepth: 1

MiV-Shared-Docs/overview/index
```
