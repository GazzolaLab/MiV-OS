# RC–KT full-cohort reproducibility workflow

These programs document the publication analysis for preflight characterization,
reservoir computing, multiscale GPFA, and knowledge transplant. They process every
manifest-approved recording. There is no quick/full switch and real data are not
distributed in this repository.

## Pipeline structure

The scripts use ordinary MiV graphs. The RC program, for example, constructs:

```python
ephys >> bandpass >> spikes
stimulus >> ttl_decoder
spikes >> trializer
ttl_decoder >> trializer
trializer >> exponential_encoder >> ridge_readout
Pipeline(ridge_readout).run(output_dir, cache_dir)
```

`batch_spontaneous.py` branches spike detection into BAKS/activity QC, branching
ratio, shuffled-ISI transfer entropy, and encoded-state kernel-rank/spectral-radius
analysis. `KT.py` pairs expert and student responses to the shared experimental
input and keeps trial selection, expert GPFA, frozen-kernel student GPFA, latent
projection, expert readout, and transplant as explicit MiV nodes with independent
caches and callbacks. The terminal transplant result supports immediate transfer,
scratch comparison, and prior-centered refinement.

## Data manifest

Copy `manifest.example.json` to `data/manifest.json` and validate it against
`manifest.schema.json`. The manifest—not filenames—is authoritative for recording
roles and expert/student pairing.

```text
data/
├── manifest.json
├── spontaneous/
├── rc_12hrs/
├── kt_expert_candidates/
└── students/
```

Every entry records its HDF5 path, source format, network/cohort/category,
elapsed time, role, stimulus channel, channel map, explicit pairing, and inclusion
status. Raw source paths are used only by the conversion program. Data, caches,
plots, result tables, and site-specific cluster configuration remain untracked.

## Conversion

`convert_data_to_h5.py` accepts manifest batches or the individual
`--source-path`, `--source-format`, and `--output-path` arguments. It streams Intan
RHS or Open Ephys recordings into aligned `Ephys` and `Stimulus` groups, preserves
the complete 30 kHz signal, and validates channel count, sample rate, timestamps,
TTL range, and MiV readability.

## Running

All programs accept `--data-dir`, `--output-dir`, `--seed`, and `--n-jobs`.
The default seed is zero. A non-cluster smoke run uses one process:

```bash
python batch_spontaneous.py --data-dir data --output-dir results --seed 0 --n-jobs 1
python RC.py --data-dir data --output-dir results --seed 0 --n-jobs 1
python KT.py --data-dir data --output-dir results --seed 0 --n-jobs 1
```

The Slurm launchers request one 32-node, 48-hour allocation on partition `normal`,
with one four-core worker per node. Parsl attaches workers to those allocated nodes;
it does not call `sbatch` or request a nested allocation. Set `RC_KT_VENV`,
`RC_KT_DATA_DIR`, `RC_KT_OUTPUT_DIR`, and optionally `RC_KT_WORKER_INIT` for the
site environment. Allocation nodes must permit worker launch over SSH.

Each program writes a `summary.json`, an inspectable cohort CSV, per-recording or
per-pair diagnostics, runtime and failure records, and headline metrics. A failed
recording is logged without cancelling independent cohort work.

## Declared analysis conventions

- readout evaluation uses a stratified 70/30 held-out split;
- GPFA uses 5 ms bins and three latent dimensions per timescale block;
- fast, task, and slow timescale bounds are 0.01–0.02 s, 0.1–2 s, and 2–100 s;
- each timescale initializes at its interval’s geometric midpoint.

Those bounds are implementation conventions derived from the reported frequency
bands, not explicit numerical values in the supporting information. They are stored
in every GPFA result and must be reported when regenerating the manuscript figures.
