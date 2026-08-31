# Core overview

**`miv.core`** is the graph layer: **lazy** linking with **`>>`**, **`output()`**, **cache policies**, and **`Pipeline`** orchestration. This section translates the ideas in [About MiV-OS](../about.md) into the objects you use in code. It is split into short pages so you can read what you need without one long document.

## Why represent an analysis as a graph?

Suppose one cleaned signal feeds spike detection, a visualization, and a
quality check. In a procedural script, reuse is implicit: it depends on the
order of statements and the variables still in scope. In MiV-OS, reuse is part
of the model. One upstream node can feed several downstream nodes, and each
terminal result tells the pipeline which dependencies it needs.

This gives the framework three useful boundaries:

- **nodes** describe data sources or scientific transformations;
- **edges** describe dependencies and shared paths; and
- **the pipeline** chooses terminal outcomes and coordinates their execution.

Caching, callbacks, streaming, and runners build on those boundaries. They do
not change what the analysis means; they change how its graph is operated.

The next pages introduce terms such as **`EagerOpNodeBase`**, **upstream**, and
**cache policy** where they first become useful. Prefer **`*NodeBase`** imports
in new code; legacy **`*Mixin`** names refer to the same classes.

---

## What to read next

| Page | Contents |
| ---- | -------- |
| **[Quickstart](core_quickstart.md)** | Lazy graphs, **`Pipeline.run`**, minimal **`EagerOpNodeBase`** example, **`tag`**. |
| **[Data types & loaders](core_datatypes_loaders.md)** | **`Signal`**, **`Spikestamps`**, **`Events`**, and **`SourceNodeBase`** / **`load()`**. |
| **[Pipeline behavior & hooks](core_pipeline_hooks.md)** | Multiple **upstream** nodes, **`output()`** vs **`__call__`**, **`Pipeline.run`** options, callbacks & plotting. |
| **[Advanced core](core_advanced.md)** | **`StreamOpNodeBase`**, generators vs **`Pipeline.run`**, cache / JSON fields, temp dirs, runners. |

---

## The smallest useful shape

```{mermaid}
graph LR
    Data("Data or source node")
    Op("Transformation")
    End("Requested result")
    Data -- ">>" --> Op
    Op -- ">>" --> End
```

---

## Other tutorials

- **[Spike sorting](spike_sorting.md)** — longer exercise using **`EagerOpNodeBase`**.
- **[MPI support](mpi_support.md)** — optional MPI / IO notes.
