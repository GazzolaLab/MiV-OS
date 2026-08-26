#!/usr/bin/env python3
"""Run full-cohort spontaneous-activity characterization (Figure 2)."""

from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path
from typing import Any

import numpy as np

try:
    from .common import (
        eligible_recordings,
        load_manifest,
        resolve_recording_path,
        run_tasks,
        write_summary,
    )
except ImportError:  # direct script execution
    from common import (
        eligible_recordings,
        load_manifest,
        resolve_recording_path,
        run_tasks,
        write_summary,
    )


def analyze_recording(task: dict[str, Any]) -> dict[str, Any]:
    entry = task["entry"]
    output = Path(task["output_dir"]) / "batch_spontaneous" / entry["id"]
    cache = output / "cache"
    started = time.time()
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        from miv.core import Pipeline
        from miv.io import ImportSignal
        from miv.signal.filter import ButterBandpass
        from miv.signal.spike import ThresholdCutoff
        from miv.statistics import (
            BayesianAdaptiveKernelSmoother,
            ExponentialSpikeEncoder,
            FixedDurationTrializer,
            KernelRank,
            SpectralRadius,
        )
        from miv.statistics.connectivity import DirectedConnectivity
        from miv.statistics.criticality import BranchingRatio

        source = ImportSignal(task["path"], group="Ephys", tag=entry["id"])
        bandpass = ButterBandpass(lowcut=300, highcut=3_000, order=4)
        spikes = ThresholdCutoff()
        baks = BayesianAdaptiveKernelSmoother(sample_rate=100.0)
        branching = BranchingRatio(bin_size=0.002)
        connectivity = DirectedConnectivity(
            bin_size=0.002,
            surrogate_N=100,
            seed=task["seed"],
        )
        trials = FixedDurationTrializer(trial_duration=1.0)
        states = ExponentialSpikeEncoder(decay_rate=5.0, sample_rate=100.0)
        rank = KernelRank(cutoff=0.99)
        radius = SpectralRadius(random_state=task["seed"])

        source >> bandpass >> spikes
        spikes >> baks
        spikes >> branching
        spikes >> connectivity
        spikes >> trials >> states
        states >> rank
        states >> radius
        Pipeline([baks, branching, connectivity, rank, radius]).run(
            output, cache, skip_plot=False, verbose=1
        )

        baks_result = baks.output()
        branching_result = branching.output()
        connectivity_result = connectivity.output()
        rank_result = rank.output()
        radius_result = radius.output()
        metrics = {
            "mean_firing_rate": float(np.nanmean(baks_result.firing_rates)),
            "branching_ratio": branching_result.ratio,
            "kernel_rank": rank_result.rank,
            "normalized_kernel_rank": rank_result.normalized_rank,
            "spectral_radius": radius_result.spectral_radius,
            "transfer_entropy_mean": float(
                np.mean(connectivity_result.transfer_entropy)
            ),
            "significant_connection_ratio": connectivity_result.connection_ratio,
        }
        output.mkdir(parents=True, exist_ok=True)
        fig, axis = plt.subplots(figsize=(9, 4))
        axis.bar(metrics.keys(), metrics.values())
        axis.tick_params(axis="x", rotation=45)
        axis.set_title(f"Preflight characterization: {entry['id']}")
        fig.tight_layout()
        fig.savefig(output / "preflight_characterization.png", dpi=180)
        plt.close(fig)
        return {
            "id": entry["id"],
            "status": "ok",
            "runtime_seconds": time.time() - started,
            "headline_metrics": metrics,
            "network_id": entry["network_id"],
            "cohort": entry["cohort"],
            "category": entry["category"],
            "elapsed_hours": entry["elapsed_hours"],
        }
    except Exception as error:
        return {"id": entry["id"], "status": "failed", "error": repr(error)}


def _write_table(output_dir: Path, results: list[dict[str, Any]]) -> None:
    rows = [item for item in results if item["status"] == "ok"]
    target = output_dir / "batch_spontaneous" / "cohort_metrics.csv"
    target.parent.mkdir(parents=True, exist_ok=True)
    metric_names = sorted({key for row in rows for key in row["headline_metrics"]})
    fields = ["id", "network_id", "cohort", "category", "elapsed_hours", *metric_names]
    with target.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {**{key: row[key] for key in fields[:5]}, **row["headline_metrics"]}
            )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--output-dir", default="results")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-jobs", type=int, default=1)
    args = parser.parse_args()
    started = time.time()
    manifest = load_manifest(args.data_dir)
    entries = eligible_recordings(manifest, {"spontaneous"})
    tasks = [
        {
            "entry": entry,
            "path": str(resolve_recording_path(args.data_dir, entry)),
            "output_dir": args.output_dir,
            "seed": args.seed,
        }
        for entry in entries
    ]
    results = run_tasks(
        analyze_recording,
        tasks,
        args.n_jobs,
        Path(args.output_dir) / "batch_spontaneous" / "parsl",
    )
    _write_table(Path(args.output_dir), results)
    write_summary(
        args.output_dir, "batch_spontaneous", vars(args), started, results
    )


if __name__ == "__main__":
    main()
