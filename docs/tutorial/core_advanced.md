# Core advanced: streaming, cache, and runners

Read this after **[Quickstart](core_quickstart.md)** and **[Pipeline behavior & hooks](core_pipeline_hooks.md)** when you use **`StreamOpNodeBase`**, hit cache surprises, or swap runners.

---

## 1. Streaming (`StreamOpNodeBase`) — eager sink or manual iteration

**`StreamOpNodeBase.output()`** can return a **generator** when not fully served from cache. **`Pipeline.run`** calls **`sink.output()`** once and **does not** exhaust that generator. If nothing iterates the result, **little or no work may run**.

**Recommended:** end the chain with an **`EagerOpNodeBase`** that **consumes** the upstream stream (e.g. **`list(...)`** or a loop), as in the core tests.

```python
# Illustrative pattern — names are placeholders
# loader_or_stream >> StreamOpOp() >> CollectAll()  # CollectAll is EagerOpNodeBase
# Pipeline(CollectAll()).run(...)
```

If the **terminal** node is streaming, **`output()`** and **iterate** (or **`list(...)`)** yourself, or insert an eager collector.

When **not** using cache, **`StreamOpNodeBase`** expects **at least one upstream** iterable on the streaming path (see implementation **`assert`**).

---

## 2. Decorators: `@cache_call` and deprecated `@cache_generator_call`

- **`@cache_call`** (**`miv.core.operator.wrapper`**) — optional eager **`__call__`** memoization in some modules.
- **`@cache_generator_call`** — **deprecated** (signature shim). Prefer **`StreamOpNodeBase`**’s **`__call__(idx, *chunk_args)`** style; see **`CONTEXT.md`** and **`operator_generator`** sources.

---

## 3. Dataclasses and JSON cache config

**`EagerOpNodeBase`** / **`StreamOpNodeBase`** use **`DataclassCacher`**: fields feed JSON **config** for cache identity. Values must be **JSON-serializable** (or **`to_json`** where supported). Putting arbitrary numpy arrays or objects in fields can make **`save_config`** raise **`TypeError`**.

Keep cache-key parameters as scalars, strings, paths, or nested JSON-friendly structures.

---

## 4. `temporary_directory` and heavy I/O

If **`temporary_directory`** is set, nodes may write under that tree during **`run`**, then **`Pipeline`** **copies** into **`working_directory`**. **`cache_directory`** follows **`set_save_path`** rules. Typical use: **MPI** or slow shared filesystem — write locally first, then aggregate (see **`Pipeline.run`** docstring).

---

## 5. Runner choice (streaming)

- **`VanillaGeneratorRunner`** (default) — matches **`StreamOpNodeBase`** **tee + zip** chunking and per-chunk **`generator_plot_*`** / cache behavior.
- **`GeneratorRunnerInMultiprocessing`** — different batching; **not** the same per-chunk alignment contract. Read **`miv.core.operator_generator.policy`** before swapping.

---

## 6. MPI and eager runners

**`VanillaRunner`**, **`StrictMPIRunner`**, **`SupportMPIMerge`** apply to **eager** operators. **`VanillaGeneratorRunner`** differs from **`VanillaRunner`** under MPI; read module docstrings before mixing MPI with streaming. User-facing MPI notes: **[MPI support](mpi_support.md)**.

---

## See also

- **`CONTEXT.md`** — glossary and **`miv.core` map**.
- **[Core overview](core_concept.md)** — diagram and section index.
