# Core: built-in data nodes and source loaders

## Built-in data nodes

Most of the time you **do not** subclass **`DataNodeBase`**. Use the library types that already inherit it:

- **`Signal`** — continuous **[time × channels]** plus timestamps and sampling rate.
- **`Spikestamps`** — per-channel spike times.
- **`Events`** — discrete times; can be binned toward **`Signal`**-like use.

They participate in **`>>`** like any other node. See the API docs for **`miv.core.datatype`** for construction details.

---

## Source nodes (`SourceNodeBase`)

Subclass **`SourceNodeBase`** for IO that **feeds** the graph from disk or devices.

- Implement **`load(*args, **kwargs)`** (often a **generator** of chunks).
- **`output()`** delegates to **`load(**_load_param)`** — not **`__call__`** (unlike operators).
- **`configure_load(**kwargs)`** stores kwargs for the next **`load`** (experimental in code; behavior may evolve).

```python
from miv.core import SourceNodeBase


class MyLoader(SourceNodeBase):
    tag: str = "my_loader"

    def __init__(self) -> None:
        super().__init__()

    def load(self, *args, **kwargs):
        yield {"chunk": 0}
        yield {"chunk": 1}
```

If **`load`** is a generator, downstream **eager** operators must match that streaming shape — see **[Advanced core](core_advanced.md)**.

---

## Next

- **[Pipeline behavior & hooks](core_pipeline_hooks.md)** — **`receive()`**, **`Pipeline.run`**, callbacks.
- **[Advanced core](core_advanced.md)** — streaming **`StreamOpNodeBase`** and generator sinks.
