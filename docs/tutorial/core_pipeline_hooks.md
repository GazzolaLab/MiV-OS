# Core: pipeline behavior, multiple upstreams, and hooks

## Multiple upstreams and `__call__` arity

**`receive()`** returns **`[u1.output(), u2.output(), …]`** in the order upstream links were **attached**. The default eager **runner** calls **`__call__(*that_list)`** — your **`__call__`** must accept **one argument per upstream**, in **that order**.

Use the same **`Literal`** / **`EagerOpNodeBase`** pattern as in **[Quickstart](core_quickstart.md)**.

```python
from dataclasses import dataclass

from miv.core import EagerOpNodeBase
from miv.core.pipeline import Pipeline


@dataclass
class Literal(EagerOpNodeBase):
    tag: str
    value: int

    def __post_init__(self) -> None:
        super().__init__()

    def __call__(self) -> int:
        return self.value


@dataclass
class Sum(EagerOpNodeBase):
    tag: str = "sum"

    def __post_init__(self) -> None:
        super().__init__()

    def __call__(self, a: int, b: int) -> int:
        return a + b


left = Literal(tag="left", value=10)
right = Literal(tag="right", value=32)
s = Sum()
left >> s
right >> s
# s receives (10, 32) because `left` was linked before `right`

Pipeline(s).run(working_directory="./results_sum", verbose=0)
assert s.output() == 42
```

- **One** upstream → **`__call__(self, x)`**.
- **Zero** upstream → **`__call__(self)`** (root nodes).

---

## `output()` vs `__call__`

- **`__call__`** — your transform (inputs → result).
- **`output()`** — framework entry: cache, **runner**, **`persist_cacher_result`**, callbacks. You normally run **`Pipeline.run`** or **`output()`** on a node; you rarely call **`__call__`** directly.

---

## `Pipeline.run`

```python
Pipeline(sink).run(
    working_directory="./results",
    cache_directory=None,       # default: same as working_directory
    temporary_directory=None,    # stage here, then copy to working_directory (see Advanced)
    skip_plot=False,
    verbose=1,
)
```

- **`skip_plot`:** skip **eager** **`plot_*`** hooks for that run (streaming hooks differ — below).
- **`Pipeline`** may take a **list** of sinks; each gets **`output()`**.

---

## Callbacks and plotting

### Eager operators (`EagerOpNodeBase`)

- **`after_run_*`** — after a successful compute path (behavior when cache hits — check code if you rely on this).
- **`plot_*`** — **`(output, inputs, show=False, save_path=None)`**; suppress with **`skip_plot=True`** on **`Pipeline.run`**.

### Streaming operators (`StreamOpNodeBase`)

- **`generator_plot_*`** — each yielded chunk.
- **`firstiter_plot_*`** — first chunk only.

Attach functions with **`<<`**; prefixes still select when they run.

### Standalone **`plot()`**

**`EagerOpNodeBase.plot(...)`** requires **cached** results (run **`Pipeline`** or **`output()`** first); otherwise **`NotImplementedError`**.

---

## Next

- **[Advanced core](core_advanced.md)** — **`StreamOpNodeBase`**, draining generators, JSON cache fields, runners.
