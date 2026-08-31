#!/usr/bin/env python3
"""Run hierarchical GPFA and knowledge-transplant analysis (Figure 5)."""

from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path
from typing import Any

import numpy as np

try:
    from .common import load_manifest, resolve_recording_path, run_tasks, write_summary
except ImportError:  # direct script execution
    from common import load_manifest, resolve_recording_path, run_tasks, write_summary


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
            GPFALatentProjector,
            KnowledgeTransfer,
            KnowledgeTransferInput,
            KnowledgeTransferInputBuilder,
            KnowledgeTransferTrialSelector,
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
            ephys >> bandpass >> spikes
            stimulus >> decoder
            spikes >> trials
            decoder >> trials
            return trials

        expert_trials = graph(expert_entry, "expert")
        student_trials = graph(student_entry, "student")
        expert_gpfa = HierarchicalGPFA(random_state=task["seed"])
        student_gpfa = HierarchicalGPFA(random_state=task["seed"])
        paired_input = KnowledgeTransferInputBuilder()
        expert_trial_stream = KnowledgeTransferTrialSelector("expert")
        student_trial_stream = KnowledgeTransferTrialSelector("student")
        expert_projector = GPFALatentProjector(9, "expert")
        student_projector = GPFALatentProjector(9, "student")
        expert_readout_node = RidgeReadout(random_state=task["seed"])
        transplant = KnowledgeTransfer(band_dimensions=(3, 3, 3))

        expert_trials >> paired_input
        student_trials >> paired_input
        paired_input >> expert_trial_stream >> expert_gpfa
        paired_input >> student_trial_stream >> student_gpfa
        expert_gpfa >> student_gpfa
        expert_gpfa >> expert_projector
        paired_input >> expert_projector
        student_gpfa >> student_projector
        paired_input >> student_projector
        expert_projector >> expert_readout_node
        expert_projector >> transplant
        student_projector >> transplant
        expert_readout_node >> transplant
        Pipeline(transplant).run(output, cache, skip_plot=True, verbose=1)

        transfer = transplant.output()
        expert_result = expert_gpfa.output()
        student_result = student_gpfa.output()
        expert_features = expert_projector.output().readout_features
        student_features = student_projector.output().readout_features
        student_labels = transfer.labels
        expert_readout = expert_readout_node.output()
        if student_labels is None or expert_features.shape != student_features.shape:
            raise RuntimeError("knowledge transfer graph returned incomplete output")
        classes = expert_readout.classes
        if not np.all(np.isin(np.unique(student_labels), classes)):
            raise ValueError("expert and student trial labels do not match")
        train_idx, test_idx = train_test_split(
            np.arange(student_labels.size),
            test_size=0.3,
            random_state=task["seed"],
            stratify=student_labels,
        )
        immediate_predictions = expert_readout.classes[
            np.argmax(
                student_features[test_idx] @ transfer.transplanted_weights, axis=1
            )
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
            selected = train_idx[:count]
            if selected.size == 0:
                raise ValueError(
                    "no paired student trials are available for refinement"
                )
            refined = KnowledgeTransfer(band_dimensions=(3, 3, 3), prior_alpha=1.0)(
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
            "learning_curve": dict(
                zip(fractions.tolist(), refined_scores, strict=True)
            ),
        }
    except Exception as error:
        return {"id": pair_id, "status": "failed", "error": repr(error)}


def _pair_tasks(manifest: dict[str, Any], args: argparse.Namespace):
    included = {
        entry["id"]: entry for entry in manifest["recordings"] if entry["included"]
    }
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
    curve_target = target.parent / "learning_curves.csv"
    with curve_target.open("w", newline="") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=["id", "training_fraction", "balanced_accuracy"]
        )
        writer.writeheader()
        for row in results:
            if row["status"] != "ok":
                continue
            for fraction, score in row["learning_curve"].items():
                writer.writerow(
                    {
                        "id": row["id"],
                        "training_fraction": fraction,
                        "balanced_accuracy": score,
                    }
                )
    rows = [row for row in results if row["status"] == "ok"]
    if rows:
        import matplotlib.pyplot as plt

        figure, axis = plt.subplots(figsize=(7, 4))
        for row in rows:
            axis.plot(
                list(row["learning_curve"]),
                list(row["learning_curve"].values()),
                marker="o",
                label=row["id"],
            )
        axis.set_xlabel("Student training fraction")
        axis.set_ylabel("Balanced accuracy")
        axis.legend()
        figure.tight_layout()
        figure.savefig(target.parent / "cohort_learning_curves.png", dpi=180)
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
