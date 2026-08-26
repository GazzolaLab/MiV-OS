#!/usr/bin/env python3
"""Run hierarchical GPFA and knowledge-transplant analysis (Figure 5)."""

from __future__ import annotations

import argparse
import csv
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from miv.core import EagerOpNodeBase

try:
    from .common import load_manifest, resolve_recording_path, run_tasks, write_summary
except ImportError:  # direct script execution
    from common import load_manifest, resolve_recording_path, run_tasks, write_summary


@dataclass
class TrialList(EagerOpNodeBase):
    """Expose the trial spikestamps from a fixed-duration trial batch."""

    tag: str = "GPFA trial list"

    def __post_init__(self) -> None:
        super().__init__()

    def __call__(self, batch):
        return batch.trials


def _latent_features(trajectories: list[np.ndarray]) -> np.ndarray:
    features = []
    for trajectory in trajectories:
        value = np.asarray(trajectory)
        if value.ndim != 2:
            raise ValueError("GPFA trajectories must be matrices")
        if value.shape[0] == 9:
            features.append(value[:, -1])
        elif value.shape[1] == 9:
            features.append(value[-1])
        else:
            raise ValueError("expected nine-dimensional GPFA trajectories")
    return np.asarray(features)


def analyze_pair(task: dict[str, Any]) -> dict[str, Any]:
    expert_entry = task["expert"]
    student_entry = task["student"]
    pair_id = f"{expert_entry['id']}__to__{student_entry['id']}"
    output = Path(task["output_dir"]) / "KT" / pair_id
    cache = output / "cache"
    started = time.time()
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from sklearn.metrics import balanced_accuracy_score
        from sklearn.model_selection import train_test_split

        from miv.core import Pipeline
        from miv.io import ImportSignal
        from miv.signal.filter import ButterBandpass
        from miv.signal.spike import ThresholdCutoff
        from miv.statistics import (
            FixedDurationTrializer,
            KnowledgeTransfer,
            KnowledgeTransferInput,
            ReservoirStateResult,
            RidgeReadout,
            TTLPulseDecoder,
        )
        from miv_state_space import HierarchicalGPFA

        def graph(entry: dict[str, Any], prefix: str):
            ephys = ImportSignal(
                task[f"{prefix}_path"], group="Ephys", tag=f"{prefix} ephys"
            )
            stimulus = ImportSignal(
                task[f"{prefix}_path"],
                group="Stimulus",
                tag=f"{prefix} stimulus",
            )
            bandpass = ButterBandpass(
                lowcut=300, highcut=3_000, order=4, tag=f"{prefix} bandpass"
            )
            spikes = ThresholdCutoff(tag=f"{prefix} spike detection")
            decoder = TTLPulseDecoder(
                stimulus_channel=entry["stimulus_channel"] or 0,
                tag=f"{prefix} TTL decoder",
            )
            trials = FixedDurationTrializer(
                trial_duration=1.0, tag=f"{prefix} trializer"
            )
            trial_list = TrialList(tag=f"{prefix} GPFA trial list")
            ephys >> bandpass >> spikes
            stimulus >> decoder
            spikes >> trials
            decoder >> trials
            trials >> trial_list
            return trials, trial_list

        expert_trials, expert_list = graph(expert_entry, "expert")
        student_trials, student_list = graph(student_entry, "student")
        expert_gpfa = HierarchicalGPFA(random_state=task["seed"])
        student_gpfa = HierarchicalGPFA(random_state=task["seed"])
        expert_list >> expert_gpfa
        student_list >> student_gpfa
        expert_gpfa >> student_gpfa
        Pipeline(student_gpfa).run(output, cache, skip_plot=True, verbose=1)

        expert_result = expert_gpfa.output()
        student_result = student_gpfa.output()
        expert_features = _latent_features(expert_result.trajectories)
        student_features = _latent_features(student_result.trajectories)
        expert_labels = expert_trials.output().labels
        student_labels = student_trials.output().labels
        paired = min(expert_features.shape[0], student_features.shape[0])
        expert_states = ReservoirStateResult(
            states=expert_features[:, None, :],
            labels=expert_labels,
            probe_times=np.asarray([0.0]),
            channel_order=tuple(range(9)),
            decay_rate=0.0,
        )
        expert_readout = RidgeReadout(random_state=task["seed"])(expert_states)
        classes = expert_readout.classes
        if not np.all(np.isin(np.unique(student_labels), classes)):
            raise ValueError("expert and student trial labels do not match")
        transfer = KnowledgeTransfer(band_dimensions=(3, 3, 3))(
            KnowledgeTransferInput(
                expert_latent=expert_features[:paired],
                student_latent=student_features[:paired],
                expert_weights=expert_readout.weights[:-1],
            )
        )
        train_idx, test_idx = train_test_split(
            np.arange(student_labels.size),
            test_size=0.3,
            random_state=task["seed"],
            stratify=student_labels,
        )
        immediate_predictions = expert_readout.classes[
            np.argmax(student_features[test_idx] @ transfer.transplanted_weights, axis=1)
        ]
        immediate_score = balanced_accuracy_score(
            student_labels[test_idx], immediate_predictions
        )
        scratch = RidgeReadout(random_state=task["seed"])(
            ReservoirStateResult(
                states=student_features[:, None, :],
                labels=student_labels,
                probe_times=np.asarray([0.0]),
                channel_order=tuple(range(9)),
                decay_rate=0.0,
            )
        )

        fractions = np.asarray([0.1, 0.25, 0.5, 0.75, 1.0])
        refined_scores = []
        one_hot = np.eye(classes.size)[np.searchsorted(classes, student_labels)]
        for fraction in fractions:
            count = max(1, int(train_idx.size * fraction))
            selected = train_idx[train_idx < paired][:count]
            if selected.size == 0:
                raise ValueError("no paired student trials are available for refinement")
            refined = KnowledgeTransfer(
                band_dimensions=(3, 3, 3), prior_alpha=1.0
            )(
                KnowledgeTransferInput(
                    expert_latent=expert_features[selected],
                    student_latent=student_features[selected],
                    expert_weights=expert_readout.weights[:-1],
                    student_targets=one_hot[selected],
                )
            ).refined_weights
            predictions = classes[
                np.argmax(student_features[test_idx] @ refined, axis=1)
            ]
            refined_scores.append(
                balanced_accuracy_score(student_labels[test_idx], predictions)
            )

        output.mkdir(parents=True, exist_ok=True)
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        axes[0].plot(expert_result.trajectories[0].T)
        axes[0].set_title("Expert latent trajectory")
        axes[1].plot(fractions, refined_scores, marker="o", label="prior-centered")
        axes[1].axhline(immediate_score, linestyle="--", label="immediate")
        axes[1].axhline(scratch.balanced_accuracy, linestyle=":", label="scratch")
        axes[1].set_xlabel("Student training fraction")
        axes[1].set_ylabel("Balanced accuracy")
        axes[1].legend()
        fig.tight_layout()
        fig.savefig(output / "latent_and_learning_curve.png", dpi=180)
        plt.close(fig)
        return {
            "id": pair_id,
            "status": "ok",
            "runtime_seconds": time.time() - started,
            "expert_id": expert_entry["id"],
            "student_id": student_entry["id"],
            "headline_metrics": {
                "immediate_balanced_accuracy": immediate_score,
                "scratch_balanced_accuracy": scratch.balanced_accuracy,
                "refined_balanced_accuracy": refined_scores[-1],
                "expert_kernel_frozen_for_student": not student_result.kernel_parameters_learned,
            },
            "learning_curve": dict(zip(fractions.tolist(), refined_scores, strict=True)),
        }
    except Exception as error:
        return {"id": pair_id, "status": "failed", "error": repr(error)}


def _pair_tasks(manifest: dict[str, Any], args: argparse.Namespace):
    included = {entry["id"]: entry for entry in manifest["recordings"] if entry["included"]}
    tasks = []
    for student in included.values():
        if student["role"] != "student":
            continue
        expert_id = student.get("expert_id")
        if not expert_id or expert_id not in included:
            raise ValueError(f"student {student['id']} has no included expert_id")
        expert = included[expert_id]
        if expert["role"] != "expert_candidate":
            raise ValueError(f"{expert_id} is not an expert_candidate")
        tasks.append(
            {
                "expert": expert,
                "student": student,
                "expert_path": str(resolve_recording_path(args.data_dir, expert)),
                "student_path": str(resolve_recording_path(args.data_dir, student)),
                "output_dir": args.output_dir,
                "seed": args.seed,
            }
        )
    return tasks


def _write_table(output_dir: Path, results: list[dict[str, Any]]) -> None:
    target = output_dir / "KT" / "knowledge_transplant.csv"
    target.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "id",
        "expert_id",
        "student_id",
        "immediate_balanced_accuracy",
        "scratch_balanced_accuracy",
        "refined_balanced_accuracy",
        "expert_kernel_frozen_for_student",
    ]
    with target.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in results:
            if row["status"] == "ok":
                writer.writerow(
                    {
                        **{key: row[key] for key in fields[:3]},
                        **row["headline_metrics"],
                    }
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
    results = run_tasks(
        analyze_pair,
        _pair_tasks(manifest, args),
        args.n_jobs,
        Path(args.output_dir) / "KT" / "parsl",
    )
    _write_table(Path(args.output_dir), results)
    write_summary(args.output_dir, "KT", vars(args), started, results)


if __name__ == "__main__":
    main()
