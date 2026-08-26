"""Reservoir-computing operators for electrophysiology analysis."""

from __future__ import annotations

from dataclasses import dataclass, field
import os
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray
from scipy.linalg import block_diag
from sklearn.metrics import balanced_accuracy_score, confusion_matrix
from sklearn.model_selection import StratifiedKFold, train_test_split

from miv.core import EagerOpNodeBase, Signal, Spikestamps
from miv.statistics.spiketrain_statistics import decay_spike_counts


def _finish_plot(fig, save_path, filename: str, show: bool) -> None:
    fig.tight_layout()
    if save_path is not None:
        fig.savefig(os.path.join(save_path, filename), dpi=180)
    if show:
        plt.show()
    plt.close(fig)


@dataclass
class TrialBatch:
    """Trial-relative spikes and their decoded temporal-pattern labels."""

    trials: list[Spikestamps]
    labels: NDArray[np.int_]
    starts: NDArray[np.float64]
    duration: float
    channel_order: tuple[int, ...]


@dataclass
class DecodedStimulus:
    """Decoded temporal pulse patterns and their trial start times."""

    starts: NDArray[np.float64]
    labels: NDArray[np.int_]
    pulse_times: NDArray[np.float64]
    stimulus_channel: int


@dataclass
class ReservoirStateResult:
    """Encoded reservoir states with trial and channel metadata."""

    states: NDArray[np.float64]
    labels: NDArray[np.int_]
    probe_times: NDArray[np.float64]
    channel_order: tuple[int, ...]
    decay_rate: float

    @property
    def readout_features(self) -> NDArray[np.float64]:
        return self.states[:, -1, :]


@dataclass
class KernelRankResult:
    rank: int
    normalized_rank: float
    singular_values: NDArray[np.float64]
    cutoff: float


@dataclass
class SpectralRadiusResult:
    spectral_radius: float
    weights: NDArray[np.float64]
    bias: NDArray[np.float64]
    losses: NDArray[np.float64]
    leak_rate: float
    random_state: int


@dataclass
class RidgeReadoutResult:
    weights: NDArray[np.float64]
    classes: NDArray[np.int_]
    alpha: float
    balanced_accuracy: float
    confusion_matrix: NDArray[np.int_]
    train_indices: NDArray[np.int_]
    test_indices: NDArray[np.int_]
    feature_order: tuple[int, ...]
    random_state: int

    def predict(self, features: NDArray[np.float64]) -> NDArray[np.int_]:
        design = np.column_stack([features, np.ones(features.shape[0])])
        return self.classes[np.argmax(design @ self.weights, axis=1)]


@dataclass
class KnowledgeTransferInput:
    """Paired expert/student observations of the same experimental inputs.

    Trial data drive the composite pipeline operator. The latent fields retain
    the lower-level alignment API for callers that already embedded recordings.
    """

    expert_trials: list[Spikestamps] | None = None
    student_trials: list[Spikestamps] | None = None
    labels: NDArray[np.int_] | None = None
    expert_latent: NDArray[np.float64] | None = None
    student_latent: NDArray[np.float64] | None = None
    expert_weights: NDArray[np.float64] | None = None
    student_targets: NDArray[np.float64] | None = None


@dataclass
class KnowledgeTransferResult:
    inverse_transform: NDArray[np.float64]
    transplanted_weights: NDArray[np.float64]
    refined_weights: NDArray[np.float64] | None
    band_dimensions: tuple[int, ...]
    alignment_alpha: float
    prior_alpha: float
    expert_latent: NDArray[np.float64] | None = None
    student_latent: NDArray[np.float64] | None = None
    labels: NDArray[np.int_] | None = None
    expert_readout: RidgeReadoutResult | None = None


@dataclass
class KnowledgeTransferInputBuilder(EagerOpNodeBase):
    """Pair expert and student responses recorded under one input protocol."""

    algorithm_version: str = "rc-kt-paired-input-v1"
    tag: str = "knowledge transfer input"

    def __post_init__(self) -> None:
        super().__init__()

    def __call__(
        self, expert: TrialBatch, student: TrialBatch
    ) -> KnowledgeTransferInput:
        if len(expert.trials) != len(student.trials):
            raise ValueError(
                "expert and student must contain the same number of trials"
            )
        if not np.array_equal(expert.labels, student.labels):
            raise ValueError(
                "expert and student trial labels must follow the same input sequence"
            )
        return KnowledgeTransferInput(
            expert_trials=expert.trials,
            student_trials=student.trials,
            labels=expert.labels.copy(),
        )


@dataclass
class KnowledgeTransferTrialSelector(EagerOpNodeBase):
    """Select one response stream from a paired knowledge-transfer input."""

    role: str = "expert"
    algorithm_version: str = "rc-kt-trial-selector-v1"
    tag: str = field(init=False)

    def __post_init__(self) -> None:
        if self.role not in {"expert", "student"}:
            raise ValueError("role must be 'expert' or 'student'")
        self.tag = f"{self.role} knowledge transfer trials"
        super().__init__()

    def __call__(self, inputs: KnowledgeTransferInput) -> list[Spikestamps]:
        trials = (
            inputs.expert_trials if self.role == "expert" else inputs.student_trials
        )
        if trials is None:
            raise ValueError(f"paired input does not contain {self.role} trials")
        return trials


@dataclass
class GPFALatentProjector(EagerOpNodeBase):
    """Convert GPFA trajectories into endpoint states for downstream operators."""

    latent_dimension: int = 9
    role: str = "expert"
    algorithm_version: str = "rc-kt-gpfa-projector-v1"
    tag: str = field(init=False)

    def __post_init__(self) -> None:
        if self.latent_dimension <= 0:
            raise ValueError("latent_dimension must be positive")
        if self.role not in {"expert", "student"}:
            raise ValueError("role must be 'expert' or 'student'")
        self.tag = f"{self.role} GPFA latent projector"
        super().__init__()

    def __call__(
        self, gpfa_result: Any, inputs: KnowledgeTransferInput
    ) -> ReservoirStateResult:
        if inputs.labels is None:
            raise ValueError("paired trial labels are required")
        features = _gpfa_latent_features(gpfa_result, self.latent_dimension)
        if features.shape[0] != inputs.labels.size:
            raise ValueError("GPFA trajectories must preserve paired trial order")
        return ReservoirStateResult(
            states=features[:, None, :],
            labels=np.asarray(inputs.labels).copy(),
            probe_times=np.asarray([0.0]),
            channel_order=tuple(range(self.latent_dimension)),
            decay_rate=0.0,
        )

    def plot_latent_features(self, result, inputs, show=False, save_path=None):
        fig, axis = plt.subplots(figsize=(8, 4))
        image = axis.imshow(result.readout_features, aspect="auto", cmap="coolwarm")
        axis.set_xlabel("Latent dimension")
        axis.set_ylabel("Trial")
        axis.set_title(f"{self.role.capitalize()} GPFA endpoint features")
        fig.colorbar(image, ax=axis)
        _finish_plot(fig, save_path, "latent_features.png", show)


@dataclass
class TTLPulseDecoder(EagerOpNodeBase):
    """Decode each TTL pulse cluster as a pulse-count stimulus label."""

    stimulus_duration: float = 0.9
    stimulus_channel: int = 0
    threshold: float = 0.5
    minimum_rest: float = 0.5
    algorithm_version: str = "rc-kt-ttl-decoder-v1"
    tag: str = "TTL pulse decoder"

    def __post_init__(self) -> None:
        if self.stimulus_duration <= 0:
            raise ValueError("stimulus_duration must be positive")
        if self.minimum_rest <= 0:
            raise ValueError("minimum_rest must be positive")
        super().__init__()

    def __call__(self, stimulus: Signal) -> DecodedStimulus:
        values = np.asarray(stimulus.data)
        if values.ndim == 2:
            values = values[:, self.stimulus_channel]
        active = values > self.threshold
        rising = np.flatnonzero(np.diff(active.astype(np.int8), prepend=0) == 1)
        pulse_times = np.asarray(stimulus.timestamps)[rising]
        if pulse_times.size == 0:
            raise ValueError("no rising TTL edges found")

        gaps = np.diff(pulse_times, prepend=-np.inf)
        starts = pulse_times[gaps >= self.minimum_rest]
        labels: list[int] = []
        for start in starts:
            count = int(
                np.count_nonzero(
                    (pulse_times >= start)
                    & (pulse_times < start + self.stimulus_duration)
                )
            )
            if count == 0:
                continue
            labels.append(count)
        return DecodedStimulus(
            starts=np.asarray(starts, dtype=np.float64),
            labels=np.asarray(labels, dtype=np.int_),
            pulse_times=pulse_times,
            stimulus_channel=self.stimulus_channel,
        )

    def plot_decoded_patterns(self, result, inputs, show=False, save_path=None):
        fig, axis = plt.subplots(figsize=(8, 3))
        axis.vlines(result.pulse_times, 0.0, 1.0, color="black", linewidth=1)
        axis.scatter(result.starts, np.ones_like(result.starts), c=result.labels)
        axis.set_xlabel("Time (s)")
        axis.set_yticks([])
        axis.set_title("Decoded TTL pulse-count patterns")
        _finish_plot(fig, save_path, "decoded_patterns.png", show)


@dataclass
class FixedDurationTrializer(EagerOpNodeBase):
    """Extract fixed-duration spike windows, optionally from decoded TTL starts."""

    trial_duration: float = 1.0
    stride: float | None = None
    algorithm_version: str = "rc-kt-fixed-trializer-v1"
    tag: str = "fixed duration trializer"

    def __post_init__(self) -> None:
        if self.trial_duration <= 0:
            raise ValueError("trial_duration must be positive")
        if self.stride is not None and self.stride <= 0:
            raise ValueError("stride must be positive")
        super().__init__()

    def __call__(
        self, spikes: Spikestamps, decoded: DecodedStimulus | None = None
    ) -> TrialBatch:
        if decoded is None:
            first = float(spikes.get_first_spikestamp())
            last = float(spikes.get_last_spikestamp())
            step = self.trial_duration if self.stride is None else self.stride
            starts = np.arange(first, last - self.trial_duration + step, step)
            labels = np.zeros(starts.size, dtype=np.int_)
        else:
            starts = decoded.starts
            labels = decoded.labels
        trials: list[Spikestamps] = []
        valid_labels: list[int] = []
        valid_starts: list[float] = []
        for start, label in zip(starts, labels, strict=True):
            stop = start + self.trial_duration
            relative = []
            for channel in spikes:
                selected = channel[(channel >= start) & (channel < stop)] - start
                relative.append(selected.tolist())
            trials.append(Spikestamps(relative))
            valid_labels.append(int(label))
            valid_starts.append(float(start))
        if not trials:
            raise ValueError("no fixed-duration trials could be constructed")
        return TrialBatch(
            trials=trials,
            labels=np.asarray(valid_labels, dtype=np.int_),
            starts=np.asarray(valid_starts),
            duration=self.trial_duration,
            channel_order=tuple(range(spikes.number_of_channels)),
        )

    def plot_trial_spike_counts(self, result, inputs, show=False, save_path=None):
        counts = [sum(len(channel) for channel in trial) for trial in result.trials]
        fig, axis = plt.subplots(figsize=(8, 3))
        axis.bar(np.arange(len(counts)), counts)
        axis.set_xlabel("Trial")
        axis.set_ylabel("Spike count")
        axis.set_title("Fixed-duration trial activity")
        _finish_plot(fig, save_path, "trial_spike_counts.png", show)


@dataclass
class ExponentialSpikeEncoder(EagerOpNodeBase):
    """Encode spikes with the causal kernel ``rho * exp(-rho * tau)``."""

    decay_rate: float = 5.0
    sample_rate: float = 500.0
    algorithm_version: str = "rc-kt-exponential-v1"
    tag: str = "exponential spike encoder"

    def __post_init__(self) -> None:
        if self.decay_rate <= 0 or self.sample_rate <= 0:
            raise ValueError("decay_rate and sample_rate must be positive")
        super().__init__()

    def __call__(self, batch: TrialBatch) -> ReservoirStateResult:
        probe_times = np.arange(0.0, batch.duration, 1.0 / self.sample_rate)
        states = np.zeros(
            (len(batch.trials), probe_times.size, len(batch.channel_order)),
            dtype=np.float64,
        )
        for trial_index, trial in enumerate(batch.trials):
            for channel_index, channel in enumerate(trial):
                states[trial_index, :, channel_index] = decay_spike_counts(
                    np.asarray(channel), probe_times, decay_rate=self.decay_rate
                )
        return ReservoirStateResult(
            states=states,
            labels=batch.labels.copy(),
            probe_times=probe_times,
            channel_order=batch.channel_order,
            decay_rate=self.decay_rate,
        )

    def plot_encoded_states(self, result, inputs, show=False, save_path=None):
        fig, axis = plt.subplots(figsize=(8, 4))
        for index in range(min(5, result.states.shape[0])):
            axis.plot(
                result.probe_times,
                result.states[index].mean(axis=1),
                label=f"trial {index}",
            )
        axis.set_xlabel("Trial time (s)")
        axis.set_ylabel("Mean encoded activity")
        axis.set_title("Causal exponential spike encoding")
        if result.states.shape[0] <= 5:
            axis.legend()
        _finish_plot(fig, save_path, "encoded_states.png", show)


def _state_matrix(value) -> NDArray[np.float64]:
    if isinstance(value, ReservoirStateResult):
        return value.states.reshape(-1, value.states.shape[-1])
    matrix = np.asarray(value, dtype=np.float64)
    if matrix.ndim == 3:
        matrix = matrix.reshape(-1, matrix.shape[-1])
    if matrix.ndim != 2:
        raise ValueError("state input must be a two- or three-dimensional array")
    return matrix


@dataclass
class KernelRank(EagerOpNodeBase):
    cutoff: float = 0.99
    channel_count: int | None = None
    algorithm_version: str = "rc-kt-kernel-rank-v1"
    tag: str = "kernel rank"

    def __post_init__(self) -> None:
        if not 0 < self.cutoff <= 1:
            raise ValueError("cutoff must be in (0, 1]")
        super().__init__()

    def __call__(self, states) -> KernelRankResult:
        matrix = np.nan_to_num(_state_matrix(states))
        singular = np.linalg.svd(matrix, compute_uv=False)
        total = float(singular.sum())
        rank = (
            0
            if np.isclose(total, 0)
            else int(np.searchsorted(np.cumsum(singular), self.cutoff * total) + 1)
        )
        channels = self.channel_count or matrix.shape[1]
        return KernelRankResult(rank, rank / channels, singular, self.cutoff)

    def plot_singular_values(self, result, inputs, show=False, save_path=None):
        fig, axis = plt.subplots(figsize=(6, 4))
        axis.semilogy(
            np.arange(1, result.singular_values.size + 1), result.singular_values
        )
        axis.axvline(
            result.rank, color="red", linestyle="--", label=f"rank={result.rank}"
        )
        axis.set_xlabel("Component")
        axis.set_ylabel("Singular value")
        axis.set_title("Kernel-rank spectrum")
        axis.legend()
        _finish_plot(fig, save_path, "kernel_rank_spectrum.png", show)


@dataclass
class SpectralRadius(EagerOpNodeBase):
    """Fit the residual echo map and report the radius of its recurrent matrix."""

    leak_rate: float = 0.9
    learning_rate: float = 1.0e-2
    epochs: int = 300
    random_state: int = 0
    algorithm_version: str = "rc-kt-spectral-radius-v1"
    tag: str = "spectral radius"

    def __post_init__(self) -> None:
        if not 0 < self.leak_rate <= 1:
            raise ValueError("leak_rate must be in (0, 1]")
        super().__init__()

    def __call__(self, states) -> SpectralRadiusResult:
        if isinstance(states, ReservoirStateResult):
            x = states.states[:, :-1, :].reshape(-1, states.states.shape[-1])
            y = states.states[:, 1:, :].reshape(-1, states.states.shape[-1])
        else:
            matrix = _state_matrix(states)
            x, y = matrix[:-1], matrix[1:]
        if x.shape[0] < 2:
            raise ValueError("at least three state samples are required")
        scale = np.maximum(np.std(x, axis=0), 1.0e-12)
        x = x / scale
        y = y / scale
        rng = np.random.default_rng(self.random_state)
        weights = rng.normal(0.0, 1.0 / np.sqrt(x.shape[1]), (x.shape[1], x.shape[1]))
        bias = np.zeros(x.shape[1])
        mw = np.zeros_like(weights)
        vw = np.zeros_like(weights)
        mb = np.zeros_like(bias)
        vb = np.zeros_like(bias)
        losses = np.zeros(self.epochs)
        for step in range(1, self.epochs + 1):
            activation = x @ weights.T - bias
            nonlinear = np.tanh(activation)
            prediction = (1 - self.leak_rate) * x + self.leak_rate * nonlinear
            error = prediction - y
            losses[step - 1] = np.mean(error**2)
            grad_activation = (
                2.0 * self.leak_rate * error * (1.0 - nonlinear**2) / error.size
            )
            grad_w = grad_activation.T @ x
            grad_b = -grad_activation.sum(axis=0)
            mw = 0.9 * mw + 0.1 * grad_w
            vw = 0.999 * vw + 0.001 * grad_w**2
            mb = 0.9 * mb + 0.1 * grad_b
            vb = 0.999 * vb + 0.001 * grad_b**2
            mw_hat = mw / (1 - 0.9**step)
            vw_hat = vw / (1 - 0.999**step)
            mb_hat = mb / (1 - 0.9**step)
            vb_hat = vb / (1 - 0.999**step)
            weights -= self.learning_rate * mw_hat / (np.sqrt(vw_hat) + 1.0e-8)
            bias -= self.learning_rate * mb_hat / (np.sqrt(vb_hat) + 1.0e-8)
        radius = float(np.max(np.abs(np.linalg.eigvals(weights))))
        return SpectralRadiusResult(
            radius, weights, bias, losses, self.leak_rate, self.random_state
        )

    def plot_fit_diagnostics(self, result, inputs, show=False, save_path=None):
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        axes[0].plot(result.losses)
        axes[0].set_xlabel("Epoch")
        axes[0].set_ylabel("Mean squared error")
        axes[0].set_title("Reservoir-map fit")
        eigenvalues = np.linalg.eigvals(result.weights)
        axes[1].scatter(eigenvalues.real, eigenvalues.imag, s=16)
        axes[1].set_xlabel("Real")
        axes[1].set_ylabel("Imaginary")
        axes[1].set_title(f"Eigenvalues (radius={result.spectral_radius:.3g})")
        _finish_plot(fig, save_path, "spectral_radius_fit.png", show)


def _ridge_weights(
    features: NDArray[np.float64],
    targets: NDArray[np.float64],
    alpha: float,
    prior: NDArray[np.float64] | None = None,
) -> NDArray[np.float64]:
    identity = np.eye(features.shape[1])
    rhs = features.T @ targets
    if prior is not None:
        rhs = rhs + alpha * prior
    return np.linalg.solve(features.T @ features + alpha * identity, rhs)


@dataclass
class RidgeReadout(EagerOpNodeBase):
    alphas: tuple[float, ...] = tuple(np.logspace(-6, 6, 13))
    test_size: float = 0.3
    cv: int = 10
    random_state: int = 0
    algorithm_version: str = "rc-kt-ridge-readout-v1"
    tag: str = "ridge readout"

    def __post_init__(self) -> None:
        super().__init__()

    def __call__(self, states: ReservoirStateResult) -> RidgeReadoutResult:
        features = states.readout_features
        labels = states.labels
        classes, encoded = np.unique(labels, return_inverse=True)
        indices = np.arange(labels.size)
        train_idx, test_idx = train_test_split(
            indices,
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=labels,
        )
        targets = np.eye(classes.size)[encoded]
        splits = min(self.cv, int(np.min(np.bincount(encoded[train_idx]))))
        if splits < 2:
            raise ValueError("each class needs at least two training trials")
        cv = StratifiedKFold(splits, shuffle=True, random_state=self.random_state)
        scores = []
        for alpha in self.alphas:
            fold_scores = []
            for inner_train, inner_valid in cv.split(
                features[train_idx], labels[train_idx]
            ):
                fit_idx = train_idx[inner_train]
                valid_idx = train_idx[inner_valid]
                design = np.column_stack([features[fit_idx], np.ones(fit_idx.size)])
                weights = _ridge_weights(design, targets[fit_idx], alpha)
                valid_design = np.column_stack(
                    [features[valid_idx], np.ones(valid_idx.size)]
                )
                predicted = classes[np.argmax(valid_design @ weights, axis=1)]
                fold_scores.append(
                    balanced_accuracy_score(labels[valid_idx], predicted)
                )
            scores.append(float(np.mean(fold_scores)))
        alpha = float(self.alphas[int(np.argmax(scores))])
        train_design = np.column_stack([features[train_idx], np.ones(train_idx.size)])
        weights = _ridge_weights(train_design, targets[train_idx], alpha)
        test_design = np.column_stack([features[test_idx], np.ones(test_idx.size)])
        predicted = classes[np.argmax(test_design @ weights, axis=1)]
        return RidgeReadoutResult(
            weights=weights,
            classes=classes,
            alpha=alpha,
            balanced_accuracy=float(
                balanced_accuracy_score(labels[test_idx], predicted)
            ),
            confusion_matrix=confusion_matrix(
                labels[test_idx], predicted, labels=classes
            ),
            train_indices=train_idx,
            test_indices=test_idx,
            feature_order=states.channel_order,
            random_state=self.random_state,
        )

    def plot_confusion_matrix(self, result, inputs, show=False, save_path=None):
        fig, axis = plt.subplots(figsize=(5, 4))
        image = axis.imshow(result.confusion_matrix, cmap="Blues")
        axis.set_xticks(range(result.classes.size), result.classes)
        axis.set_yticks(range(result.classes.size), result.classes)
        axis.set_xlabel("Predicted class")
        axis.set_ylabel("True class")
        axis.set_title(f"Balanced accuracy: {result.balanced_accuracy:.3f}")
        fig.colorbar(image, ax=axis)
        _finish_plot(fig, save_path, "confusion_matrix.png", show)


def _gpfa_latent_features(result: Any, dimension: int) -> NDArray[np.float64]:
    """Extract one aligned endpoint feature vector from each GPFA trajectory."""

    features = []
    for trajectory in result.trajectories:
        value = np.asarray(trajectory, dtype=np.float64)
        if value.ndim != 2:
            raise ValueError("GPFA trajectories must be two-dimensional")
        if value.shape[0] == dimension:
            features.append(value[:, -1])
        elif value.shape[1] == dimension:
            features.append(value[-1])
        else:
            raise ValueError(
                f"expected GPFA trajectories with {dimension} latent dimensions"
            )
    return np.asarray(features, dtype=np.float64)


@dataclass
class KnowledgeTransfer(EagerOpNodeBase):
    """Align paired latent states and transplant an already fitted readout.

    This operator deliberately does not fit GPFA or readout modules internally.
    Those remain explicit upstream MiV nodes with independent caches and
    callbacks. A single :class:`KnowledgeTransferInput` retains the lower-level
    pre-embedded array API.
    """

    band_dimensions: tuple[int, ...] = (3, 3, 3)
    alignment_alpha: float = 1.0e-6
    prior_alpha: float = 1.0
    refine: bool = True
    algorithm_version: str = "rc-kt-knowledge-transfer-v1"
    tag: str = "knowledge transfer"

    def __post_init__(self) -> None:
        if any(size <= 0 for size in self.band_dimensions):
            raise ValueError("band dimensions must be positive")
        super().__init__()

    def __call__(
        self,
        expert_input: KnowledgeTransferInput | ReservoirStateResult,
        student_states: ReservoirStateResult | None = None,
        expert_readout: RidgeReadoutResult | None = None,
    ) -> KnowledgeTransferResult:
        dimension = sum(self.band_dimensions)

        if isinstance(expert_input, KnowledgeTransferInput):
            inputs = expert_input
            if (
                inputs.expert_latent is None
                or inputs.student_latent is None
                or inputs.expert_weights is None
            ):
                raise ValueError(
                    "latent inputs and expert_weights are required without GPFA modules"
                )
            expert = np.asarray(inputs.expert_latent, dtype=np.float64)
            student = np.asarray(inputs.student_latent, dtype=np.float64)
            expert_weights = np.asarray(inputs.expert_weights, dtype=np.float64)
            targets = inputs.student_targets
            labels = inputs.labels
            readout_result = None
        else:
            if student_states is None or expert_readout is None:
                raise ValueError(
                    "expert states, student states, and expert readout are required"
                )
            if not np.array_equal(expert_input.labels, student_states.labels):
                raise ValueError("expert and student states must retain paired labels")
            expert = expert_input.readout_features
            student = student_states.readout_features
            expert_weights = expert_readout.weights[:-1]
            labels = student_states.labels
            readout_result = expert_readout
            targets = None
            if self.refine:
                class_indices = {
                    value: index for index, value in enumerate(expert_readout.classes)
                }
                try:
                    encoded = np.asarray([class_indices[value] for value in labels])
                except KeyError as error:
                    raise ValueError(
                        "student labels must be represented in the expert readout"
                    ) from error
                targets = np.eye(expert_readout.classes.size)[encoded]

        if expert.shape != student.shape:
            raise ValueError("paired expert and student latents must have equal shape")
        if expert.shape[1] != dimension:
            raise ValueError("band dimensions must cover the latent dimensions")
        transforms = []
        offset = 0
        for size in self.band_dimensions:
            zs = student[:, offset : offset + size]
            ze = expert[:, offset : offset + size]
            transforms.append(_ridge_weights(zs, ze, self.alignment_alpha))
            offset += size
        inverse_transform = block_diag(*transforms)
        transplanted = inverse_transform @ expert_weights
        refined = None
        if targets is not None:
            refined = _ridge_weights(
                student,
                np.asarray(targets),
                self.prior_alpha,
                prior=transplanted,
            )
        return KnowledgeTransferResult(
            inverse_transform=inverse_transform,
            transplanted_weights=transplanted,
            refined_weights=refined,
            band_dimensions=self.band_dimensions,
            alignment_alpha=self.alignment_alpha,
            prior_alpha=self.prior_alpha,
            expert_latent=expert,
            student_latent=student,
            labels=None if labels is None else np.asarray(labels).copy(),
            expert_readout=readout_result,
        )

    def plot_alignment(self, result, inputs, show=False, save_path=None):
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        image = axes[0].imshow(result.inverse_transform, cmap="coolwarm")
        axes[0].set_title("Block-diagonal latent alignment")
        axes[0].set_xlabel("Expert latent dimension")
        axes[0].set_ylabel("Student latent dimension")
        fig.colorbar(image, ax=axes[0])
        axes[1].plot(result.expert_latent[:, : min(3, result.expert_latent.shape[1])])
        axes[1].set_xlabel("Paired trial")
        axes[1].set_ylabel("Expert latent endpoint")
        axes[1].set_title("Aligned expert features")
        _finish_plot(fig, save_path, "knowledge_transfer_alignment.png", show)


__all__ = [
    "DecodedStimulus",
    "ExponentialSpikeEncoder",
    "FixedDurationTrializer",
    "GPFALatentProjector",
    "KernelRank",
    "KernelRankResult",
    "KnowledgeTransfer",
    "KnowledgeTransferInput",
    "KnowledgeTransferInputBuilder",
    "KnowledgeTransferResult",
    "KnowledgeTransferTrialSelector",
    "ReservoirStateResult",
    "RidgeReadout",
    "RidgeReadoutResult",
    "SpectralRadius",
    "SpectralRadiusResult",
    "TTLPulseDecoder",
    "TrialBatch",
]
