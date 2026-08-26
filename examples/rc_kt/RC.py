#!/usr/bin/env python3
"""Run longitudinal reservoir-computing evaluation (Figures 3 and 4)."""

from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path
from typing import Any

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
    output = Path(task["output_dir"]) / "RC" / entry["id"]
    cache = output / "cache"
    started = time.time()
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np

        from miv.core import Pipeline
        from miv.io import ImportSignal
        from miv.signal.filter import ButterBandpass
        from miv.signal.spike import ThresholdCutoff
        from miv.statistics import (
            ExponentialSpikeEncoder,
            FixedDurationTrializer,
            RidgeReadout,
            TTLPulseDecoder,
        )

        ephys = ImportSignal(task["path"], group="Ephys", tag=f"{entry['id']} ephys")
        stimulus = ImportSignal(
            task["path"], group="Stimulus", tag=f"{entry['id']} stimulus"
        )
        bandpass = ButterBandpass(lowcut=300, highcut=3_000, order=4)
        spikes = ThresholdCutoff()
        decoder = TTLPulseDecoder(stimulus_channel=entry["stimulus_channel"] or 0)
        trializer = FixedDurationTrializer(trial_duration=1.0)
        encoder = ExponentialSpikeEncoder(decay_rate=5.0, sample_rate=200.0)
        readout = RidgeReadout(
            test_size=0.3,
            cv=10,
            random_state=task["seed"],
        )

        ephys >> bandpass >> spikes
        stimulus >> decoder
        spikes >> trializer
        decoder >> trializer
        trializer >> encoder >> readout
        Pipeline(readout).run(output, cache, skip_plot=True, verbose=1)

        result = readout.output()
        output.mkdir(parents=True, exist_ok=True)
        fig, axis = plt.subplots(figsize=(5, 4))
        image = axis.imshow(result.confusion_matrix, cmap="Blues")
        axis.set_xticks(range(len(result.classes)), result.classes)
        axis.set_yticks(range(len(result.classes)), result.classes)
        axis.set_xlabel("Predicted pulse count")
        axis.set_ylabel("True pulse count")
        axis.set_title(f"{entry['id']} (BA={result.balanced_accuracy:.3f})")
        fig.colorbar(image, ax=axis)
        fig.tight_layout()
        fig.savefig(output / "confusion_matrix.png", dpi=180)
        plt.close(fig)
        return {
            "id": entry["id"],
            "status": "ok",
            "runtime_seconds": time.time() - started,
            "network_id": entry["network_id"],
            "cohort": entry["cohort"],
            "elapsed_hours": entry["elapsed_hours"],
            "headline_metrics": {
                "balanced_accuracy": result.balanced_accuracy,
                "ridge_alpha": result.alpha,
                "trials_train": int(result.train_indices.size),
                "trials_test": int(result.test_indices.size),
            },
        }
    except Exception as error:
        return {"id": entry["id"], "status": "failed", "error": repr(error)}


def _write_table(output_dir: Path, results: list[dict[str, Any]]) -> None:
    target = output_dir / "RC" / "longitudinal_performance.csv"
    target.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "id",
        "network_id",
        "cohort",
        "elapsed_hours",
        "balanced_accuracy",
        "ridge_alpha",
        "trials_train",
        "trials_test",
    ]
    with target.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in results:
            if row["status"] != "ok":
                continue
            writer.writerow(
                {
                    **{key: row[key] for key in fields[:4]},
                    **row["headline_metrics"],
                }
            )
    rows = [row for row in results if row["status"] == "ok"]
    if rows:
        import matplotlib.pyplot as plt

        figure, axis = plt.subplots(figsize=(7, 4))
        for network_id in sorted({row["network_id"] for row in rows}):
            network = sorted(
                (row for row in rows if row["network_id"] == network_id),
                key=lambda row: row["elapsed_hours"],
            )
            axis.plot(
                [row["elapsed_hours"] for row in network],
                [row["headline_metrics"]["balanced_accuracy"] for row in network],
                marker="o",
                label=network_id,
            )
        axis.set_xlabel("Elapsed time (hours)")
        axis.set_ylabel("Balanced accuracy")
        axis.legend()
        figure.tight_layout()
        figure.savefig(target.parent / "longitudinal_performance.png", dpi=180)
        plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--output-dir", default="results")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-jobs", type=int, default=1)
    args = parser.parse_args()
    started = time.time()
    manifest = load_manifest(args.data_dir)
    tasks = [
        {
            "entry": entry,
            "path": str(resolve_recording_path(args.data_dir, entry)),
            "output_dir": args.output_dir,
            "seed": args.seed,
        }
        for entry in eligible_recordings(manifest, {"rc"})
    ]
    results = run_tasks(
        analyze_recording,
        tasks,
        args.n_jobs,
        Path(args.output_dir) / "RC" / "parsl",
    )
    _write_table(Path(args.output_dir), results)
    write_summary(args.output_dir, "RC", vars(args), started, results)


if __name__ == "__main__":
    main()
