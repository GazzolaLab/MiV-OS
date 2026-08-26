"""Reservoir-computing operators for electrophysiology analysis."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray
from scipy.linalg import block_diag
from sklearn.metrics import balanced_accuracy_score, confusion_matrix
from sklearn.model_selection import StratifiedKFold, train_test_split

from miv.core import EagerOpNodeBase, Signal, Spikestamps
from miv.statistics.spiketrain_statistics import decay_spike_counts


@dataclass
class TrialBatch:
    """Trial-relative spikes and their decoded temporal-pattern labels."""

    trials: list[Spikestamps]
    labels: NDArray[np.int_]
    starts: NDArray[np.float64]
    duration: float
    channel_order: tuple[int, ...]


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
    expert_latent: NDArray[np.float64]
    student_latent: NDArray[np.float64]
    expert_weights: NDArray[np.float64]
    student_targets: NDArray[np.float64] | None = None


@dataclass
class KnowledgeTransferResult:
    inverse_transform: NDArray[np.float64]
    transplanted_weights: NDArray[np.float64]
    refined_weights: NDArray[np.float64] | None
    band_dimensions: tuple[int, ...]
    alignment_alpha: float
    prior_alpha: float


@dataclass
class StimulusTrializer(EagerOpNodeBase):
    """Decode pulse-count patterns from a digital stimulus signal."""

    trial_duration: float = 1.0
    stimulus_duration: float = 0.9
    stimulus_channel: int = 0
    threshold: float = 0.5
    minimum_rest: float = 0.5
    algorithm_version: str = "rc-kt-trializer-v1"
    tag: str = field(default="stimulus trializer", init=False)

    def __post_init__(self) -> None:
        if not 0 < self.stimulus_duration <= self.trial_duration:
            raise ValueError("stimulus_duration must be within the trial")
        if self.minimum_rest <= 0:
            raise ValueError("minimum_rest must be positive")
        super().__init__()

    def __call__(self, spikes: Spikestamps, stimulus: Signal) -> TrialBatch:
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
        trials: list[Spikestamps] = []
        labels: list[int] = []
        valid_starts: list[float] = []
        for start in starts:
            stop = start + self.trial_duration
            count = int(
                np.count_nonzero(
                    (pulse_times >= start)
                    & (pulse_times < start + self.stimulus_duration)
                )
            )
            if count == 0:
                continue
            relative = []
            for channel in spikes:
                selected = channel[(channel >= start) & (channel < stop)] - start
                relative.append(selected.tolist())
            trials.append(Spikestamps(relative))
            labels.append(count)
            valid_starts.append(float(start))
        if not trials:
            raise ValueError("TTL edges did not produce any complete trials")
        return TrialBatch(
            trials=trials,
            labels=np.asarray(labels, dtype=np.int_),
            starts=np.asarray(valid_starts),
            duration=self.trial_duration,
            channel_order=tuple(range(spikes.number_of_channels)),
        )


@dataclass
class ExponentialSpikeEncoder(EagerOpNodeBase):
    """Encode spikes with the causal kernel ``rho * exp(-rho * tau)``."""

    decay_rate: float = 5.0
    sample_rate: float = 500.0
    algorithm_version: str = "rc-kt-exponential-v1"
    tag: str = field(default="exponential spike encoder", init=False)

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
    tag: str = field(default="kernel rank", init=False)

    def __post_init__(self) -> None:
        if not 0 < self.cutoff <= 1:
            raise ValueError("cutoff must be in (0, 1]")
        super().__init__()

    def __call__(self, states) -> KernelRankResult:
        matrix = np.nan_to_num(_state_matrix(states))
        singular = np.linalg.svd(matrix, compute_uv=False)
        total = float(singular.sum())
        rank = 0 if np.isclose(total, 0) else int(
            np.searchsorted(np.cumsum(singular), self.cutoff * total) + 1
        )
        channels = self.channel_count or matrix.shape[1]
        return KernelRankResult(rank, rank / channels, singular, self.cutoff)


@dataclass
class SpectralRadius(EagerOpNodeBase):
    """Fit the residual echo map and report the radius of its recurrent matrix."""

    leak_rate: float = 0.9
    learning_rate: float = 1.0e-2
    epochs: int = 300
    random_state: int = 0
    algorithm_version: str = "rc-kt-spectral-radius-v1"
    tag: str = field(default="spectral radius", init=False)

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
                2.0
                * self.leak_rate
                * error
                * (1.0 - nonlinear**2)
                / error.size
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
    tag: str = field(default="ridge readout", init=False)

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
            for inner_train, inner_valid in cv.split(features[train_idx], labels[train_idx]):
                fit_idx = train_idx[inner_train]
                valid_idx = train_idx[inner_valid]
                design = np.column_stack([features[fit_idx], np.ones(fit_idx.size)])
                weights = _ridge_weights(design, targets[fit_idx], alpha)
                valid_design = np.column_stack(
                    [features[valid_idx], np.ones(valid_idx.size)]
                )
                predicted = classes[np.argmax(valid_design @ weights, axis=1)]
                fold_scores.append(balanced_accuracy_score(labels[valid_idx], predicted))
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
            confusion_matrix=confusion_matrix(labels[test_idx], predicted, labels=classes),
            train_indices=train_idx,
            test_indices=test_idx,
            feature_order=states.channel_order,
            random_state=self.random_state,
        )


@dataclass
class KnowledgeTransfer(EagerOpNodeBase):
    band_dimensions: tuple[int, ...] = (3, 3, 3)
    alignment_alpha: float = 1.0e-6
    prior_alpha: float = 1.0
    algorithm_version: str = "rc-kt-knowledge-transfer-v1"
    tag: str = field(default="knowledge transfer", init=False)

    def __post_init__(self) -> None:
        if any(size <= 0 for size in self.band_dimensions):
            raise ValueError("band dimensions must be positive")
        super().__init__()

    def __call__(self, inputs: KnowledgeTransferInput) -> KnowledgeTransferResult:
        expert = np.asarray(inputs.expert_latent, dtype=np.float64)
        student = np.asarray(inputs.student_latent, dtype=np.float64)
        if expert.shape != student.shape:
            raise ValueError("paired expert and student latents must have equal shape")
        if expert.shape[1] != sum(self.band_dimensions):
            raise ValueError("band dimensions must cover the latent dimensions")
        transforms = []
        offset = 0
        for size in self.band_dimensions:
            zs = student[:, offset : offset + size]
            ze = expert[:, offset : offset + size]
            transforms.append(_ridge_weights(zs, ze, self.alignment_alpha))
            offset += size
        inverse_transform = block_diag(*transforms)
        transplanted = inverse_transform @ inputs.expert_weights
        refined = None
        if inputs.student_targets is not None:
            refined = _ridge_weights(
                student,
                np.asarray(inputs.student_targets),
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
        )


__all__ = [
    "ExponentialSpikeEncoder",
    "KernelRank",
    "KernelRankResult",
    "KnowledgeTransfer",
    "KnowledgeTransferInput",
    "KnowledgeTransferResult",
    "ReservoirStateResult",
    "RidgeReadout",
    "RidgeReadoutResult",
    "SpectralRadius",
    "SpectralRadiusResult",
    "StimulusTrializer",
    "TrialBatch",
]
