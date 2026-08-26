# Core overview

**`miv.core`** is the graph layer: **lazy** linking with **`>>`**, **`output()`**, **cache policies**, and **`Pipeline`** orchestration. This section of the tutorial is split into short pages so you can read what you need without one long document.

**Vocabulary:** terms like **`EagerOpNodeBase`**, **upstream**, **cache policy** are defined in the repo root **`CONTEXT.md`**. Prefer **`*NodeBase`** imports in new code; legacy **`*Mixin`** names are the same classes.

---

## What to read next

| Page | Contents |
| ---- | -------- |
| **[Quickstart](core_quickstart.md)** | Lazy graphs, **`Pipeline.run`**, minimal **`EagerOpNodeBase`** example, **`tag`**. |
| **[Data types & loaders](core_datatypes_loaders.md)** | **`Signal`**, **`Spikestamps`**, **`Events`**, and **`SourceNodeBase`** / **`load()`**. |
| **[Pipeline behavior & hooks](core_pipeline_hooks.md)** | Multiple **upstream** nodes, **`output()`** vs **`__call__`**, **`Pipeline.run`** options, callbacks & plotting. |
| **[Advanced core](core_advanced.md)** | **`StreamOpNodeBase`**, generators vs **`Pipeline.run`**, cache / JSON fields, temp dirs, runners. |

---

## Reference diagram

```{mermaid}
graph LR
    Data("Data / source node")
    Op("Eager operator")
    End("Sink (often eager)")
    Data -- ">>" --> Op
    Op -- ">>" --> End
```

---

## Other tutorials

- **`CONTEXT.md`** — full glossary and **`miv.core` module map**.
- **[Spike sorting](spike_sorting.md)** — longer exercise using **`EagerOpNodeBase`**.
- **[MPI support](mpi_support.md)** — optional MPI / IO notes.
