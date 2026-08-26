# Core quickstart: lazy graphs and your first pipeline

This page is the shortest path: link nodes with **`>>`**, run **`Pipeline`**, and subclass **`EagerOpNodeBase`** once.

---

## Lazy linking

**`a >> b`** only records that **`b`** depends on **`a`**. Heavy work runs when something calls **`output()`** on a node, or when you **`Pipeline(...).run()`** (which calls **`output()`** on each **sink** you pass in).

**Eager operators** subclass **`EagerOpNodeBase`**. Each run of **`output()`** returns **one** value (often a **`DataType`**; the toy example below uses integers).

---

## Minimal example (single upstream)

```python
from dataclasses import dataclass

from miv.core import EagerOpNodeBase
from miv.core.pipeline import Pipeline


@dataclass
class Literal(EagerOpNodeBase):
    """Root node with no upstream: __call__ takes no arguments."""

    tag: str
    value: int

    def __post_init__(self) -> None:
        super().__init__()

    def __call__(self) -> int:
        return self.value


@dataclass
class Double(EagerOpNodeBase):
    tag: str = "double"

    def __post_init__(self) -> None:
        super().__init__()

    def __call__(self, x: int) -> int:
        return x * 2


lit = Literal(tag="literal", value=21)
dbl = Double()
lit >> dbl

Pipeline(dbl).run(working_directory="./results_minimal", verbose=0)
assert dbl.output() == 42
```

What happened:

1. **`lit >> dbl`** makes **`lit`** **upstream** of **`dbl`**.
2. **`Pipeline(dbl).run(...)`** walks the graph in **topological order** and eventually calls **`dbl.output()`**.
3. **`dbl.output()`** pulls **upstream** via **`receive()`** (calls **`lit.output()`**), then runs **`dbl.__call__(21)`**.

**`tag`:** use a distinct **`tag`** per node you care to cache or identify in logs (paths and **`repr`**).

---

## Next

- **[Data types & loaders](core_datatypes_loaders.md)** — built-in **`Signal`** / **`Spikestamps`** / **`Events`** and **`SourceNodeBase`**.
- **[Pipeline behavior & hooks](core_pipeline_hooks.md)** — multiple upstreams, **`Pipeline.run`** kwargs, callbacks.
