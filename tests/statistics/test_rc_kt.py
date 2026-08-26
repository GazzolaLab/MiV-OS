from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np

from miv.core import EagerOpNodeBase, Pipeline, Signal, Spikestamps
from miv.statistics import (
    BayesianAdaptiveKernelSmoother,
    ExponentialSpikeEncoder,
    FixedDurationTrializer,
    GPFALatentProjector,
    KernelRank,
    KnowledgeTransfer,
    KnowledgeTransferInput,
    KnowledgeTransferInputBuilder,
    KnowledgeTransferTrialSelector,
    ReservoirStateResult,
    RidgeReadout,
    SpectralRadius,
    TTLPulseDecoder,
)
from miv.statistics.connectivity.connectivity import (
    _discrete_transfer_entropy,
    _shuffle_interspike_intervals,
)
from miv.statistics.criticality import BranchingRatio
from miv.statistics.reservoir import TrialBatch


@dataclass(eq=False)
class _BatchSource(EagerOpNodeBase):
    batch: TrialBatch
    tag: str = "test batch source"

    def __post_init__(self) -> None:
        super().__init__()

    def __call__(self) -> TrialBatch:
        return self.batch


@dataclass(eq=False)
class _FakeGPFA(EagerOpNodeBase):
    features: np.ndarray
    tag: str

    def __post_init__(self) -> None:
        self.expert_reference: Any | None = None
        super().__init__()

    def __call__(self, trials, expert=None):
        assert len(trials) == self.features.shape[0]
        self.expert_reference = expert
        return SimpleNamespace(
            trajectories=[row[:, None] for row in self.features],
            timescales=np.ones(self.features.shape[1]),
            kernel_parameters_learned=expert is None,
        )


def test_baks_uses_beta_term() -> None:
    result = BayesianAdaptiveKernelSmoother(
        alpha=4.0, beta=1.0, sample_rate=2.0, t_start=0.0, t_end=1.0
    )(Spikestamps([[0.0]]))
    expected_h = math.gamma(4) / math.gamma(4.5)
    np.testing.assert_allclose(result.bandwidths[0, 0], expected_h)
    assert np.isfinite(result.firing_rates).all()


def test_trializer_and_exponential_encoder() -> None:
    timestamps = np.arange(0.0, 2.0, 0.01)
    ttl = np.zeros_like(timestamps)
    for pulse in (0.1, 0.2, 1.1, 1.2, 1.3):
        ttl[np.searchsorted(timestamps, pulse)] = 1.0
    stimulus = Signal(ttl[:, None], timestamps, 100.0)
    spikes = Spikestamps([[0.15, 1.15], [0.25, 1.25]])
    decoded = TTLPulseDecoder(minimum_rest=0.5)(stimulus)
    batch = FixedDurationTrializer()(spikes, decoded)
    np.testing.assert_array_equal(batch.labels, [2, 3])
    encoded = ExponentialSpikeEncoder(decay_rate=5.0, sample_rate=10.0)(batch)
    assert encoded.states.shape == (2, 10, 2)
    assert encoded.states[0, 2, 0] > 0


def test_branching_ratio_counts_valid_transitions() -> None:
    spikes = Spikestamps([[0.01, 0.11, 0.21, 0.31], [0.11, 0.21, 0.31]])
    result = BranchingRatio(bin_size=0.1)(spikes)
    assert result.valid_transitions == 2
    np.testing.assert_allclose(result.ratio, (4.0 / 1.0 + 2.0 / 4.0) / 2.0)


def test_kernel_rank_uses_cumulative_singular_value_mass() -> None:
    matrix = np.diag([4.0, 1.0, 0.1])
    result = KernelRank(cutoff=0.9)(matrix)
    assert result.rank == 2
    np.testing.assert_allclose(result.normalized_rank, 2 / 3)


def test_transfer_entropy_detects_direction() -> None:
    rng = np.random.default_rng(4)
    source = rng.integers(0, 2, 1000)
    target = np.zeros_like(source)
    target[1:] = source[:-1]
    assert _discrete_transfer_entropy(source, target) > 0.8
    assert _discrete_transfer_entropy(target, source) < 0.02


def test_shuffled_isi_bootstrap_is_seeded_and_preserves_intervals() -> None:
    source = np.zeros(40, dtype=int)
    source[[2, 5, 11, 18, 29]] = 1
    first = _shuffle_interspike_intervals(source, np.random.default_rng(7))
    second = _shuffle_interspike_intervals(source, np.random.default_rng(7))
    np.testing.assert_array_equal(first, second)
    np.testing.assert_array_equal(
        np.sort(np.diff(np.flatnonzero(first))),
        np.sort(np.diff(np.flatnonzero(source))),
    )


def test_ridge_readout_is_reproducible() -> None:
    labels = np.repeat([1, 2], 20)
    features = np.column_stack([labels == 1, labels == 2]).astype(float)
    states = ReservoirStateResult(
        states=features[:, None, :],
        labels=labels,
        probe_times=np.array([0.0]),
        channel_order=(0, 1),
        decay_rate=5.0,
    )
    first = RidgeReadout(alphas=(1.0e-6, 1.0), cv=4)(states)
    second = RidgeReadout(alphas=(1.0e-6, 1.0), cv=4)(states)
    assert first.balanced_accuracy == 1.0
    np.testing.assert_allclose(first.weights, second.weights)


def test_knowledge_transfer_and_prior_centered_limits() -> None:
    rng = np.random.default_rng(0)
    student = rng.normal(size=(100, 4))
    transform = np.diag([2.0, 2.0, 0.5, 0.5])
    expert = student @ transform
    expert_weights = rng.normal(size=(4, 2))
    targets = student @ transform @ expert_weights
    result = KnowledgeTransfer(
        band_dimensions=(2, 2), alignment_alpha=1.0e-10, prior_alpha=1.0e6
    )(
        KnowledgeTransferInput(
            expert_latent=expert,
            student_latent=student,
            expert_weights=expert_weights,
            student_targets=targets,
        )
    )
    np.testing.assert_allclose(result.inverse_transform, transform, atol=1.0e-8)
    np.testing.assert_allclose(
        result.refined_weights, result.transplanted_weights, atol=1.0e-4
    )

    unregularized = KnowledgeTransfer(
        band_dimensions=(2, 2), alignment_alpha=1.0e-10, prior_alpha=0.0
    )(
        KnowledgeTransferInput(
            expert_latent=expert,
            student_latent=student,
            expert_weights=np.zeros_like(expert_weights),
            student_targets=targets,
        )
    )
    expected, *_ = np.linalg.lstsq(student, targets, rcond=None)
    np.testing.assert_allclose(unregularized.refined_weights, expected, atol=1.0e-8)


def test_modular_knowledge_transfer_pipeline_uses_one_paired_input(tmp_path) -> None:
    labels = np.repeat([1, 2], 20)
    expert_features = np.column_stack(
        [
            labels == 1,
            labels == 2,
            0.5 * (labels == 1),
            0.5 * (labels == 2),
        ]
    ).astype(float)
    student_features = expert_features @ np.diag([0.5, 0.5, 2.0, 2.0])
    empty_trials = [Spikestamps([[]]) for _ in labels]
    expert_batch = TrialBatch(empty_trials, labels, np.arange(labels.size), 1.0, (0,))
    student_batch = TrialBatch(
        empty_trials.copy(), labels.copy(), np.arange(labels.size), 1.0, (0,)
    )
    expert_source = _BatchSource(expert_batch, tag="expert trials")
    student_source = _BatchSource(student_batch, tag="student trials")
    paired_input = KnowledgeTransferInputBuilder()
    expert_trials = KnowledgeTransferTrialSelector("expert")
    student_trials = KnowledgeTransferTrialSelector("student")
    expert_gpfa = _FakeGPFA(expert_features, "expert GPFA")
    student_gpfa = _FakeGPFA(student_features, "student GPFA")
    expert_projector = GPFALatentProjector(4, "expert")
    student_projector = GPFALatentProjector(4, "student")
    readout = RidgeReadout(alphas=(1.0e-6,), cv=2, random_state=0)
    transplant = KnowledgeTransfer(band_dimensions=(2, 2))
    nodes = (
        expert_source,
        student_source,
        paired_input,
        expert_trials,
        student_trials,
        expert_gpfa,
        student_gpfa,
        expert_projector,
        student_projector,
        readout,
        transplant,
    )
    for node in nodes:
        node.set_caching_policy("OFF")

    expert_source >> paired_input
    student_source >> paired_input
    paired_input >> expert_trials >> expert_gpfa
    paired_input >> student_trials >> student_gpfa
    expert_gpfa >> student_gpfa
    expert_gpfa >> expert_projector
    paired_input >> expert_projector
    student_gpfa >> student_projector
    paired_input >> student_projector
    expert_projector >> readout
    expert_projector >> transplant
    student_projector >> transplant
    readout >> transplant

    Pipeline(transplant).run(tmp_path, skip_plot=False)
    result = transplant.output()

    assert student_gpfa.expert_reference is not None
    np.testing.assert_array_equal(result.labels, labels)
    assert result.expert_readout is not None
    assert result.transplanted_weights.shape == (4, 2)
    assert (tmp_path / "expert_GPFA_latent_projector/latent_features.png").is_file()
    assert (tmp_path / "ridge_readout/confusion_matrix.png").is_file()
    assert (tmp_path / "knowledge_transfer/knowledge_transfer_alignment.png").is_file()


def test_standalone_operator_callbacks_write_diagnostics(tmp_path) -> None:
    timestamps = np.arange(0.0, 2.0, 0.01)
    ttl_values = np.zeros_like(timestamps)
    for pulse in (0.1, 0.2, 1.1, 1.2, 1.3):
        ttl_values[np.searchsorted(timestamps, pulse)] = 1.0
    stimulus = Signal(ttl_values[:, None], timestamps, 100.0)
    spikes = Spikestamps([[0.15, 0.25, 1.15], [0.35, 1.25]])

    decoder = TTLPulseDecoder(minimum_rest=0.5)
    decoded = decoder(stimulus)
    trializer = FixedDurationTrializer()
    trials = trializer(spikes, decoded)
    encoder = ExponentialSpikeEncoder(decay_rate=5.0, sample_rate=10.0)
    states = encoder(trials)
    baks = BayesianAdaptiveKernelSmoother(sample_rate=2.0, t_start=0.0, t_end=2.0)
    baks_result = baks(spikes)
    rank = KernelRank()
    rank_result = rank(states)
    radius = SpectralRadius(epochs=3, random_state=0)
    radius_result = radius(states)
    branching = BranchingRatio(bin_size=0.1)
    branching_result = branching(spikes)

    callbacks = (
        (decoder, decoder.plot_decoded_patterns, decoded, "decoded_patterns.png"),
        (
            trializer,
            trializer.plot_trial_spike_counts,
            trials,
            "trial_spike_counts.png",
        ),
        (encoder, encoder.plot_encoded_states, states, "encoded_states.png"),
        (baks, baks.plot_firing_rates, baks_result, "baks_firing_rates.png"),
        (rank, rank.plot_singular_values, rank_result, "kernel_rank_spectrum.png"),
        (
            radius,
            radius.plot_fit_diagnostics,
            radius_result,
            "spectral_radius_fit.png",
        ),
        (
            branching,
            branching.plot_branching_ratio,
            branching_result,
            "branching_ratio_scalar.png",
        ),
    )
    for operator, callback, result, filename in callbacks:
        operator.set_save_path(tmp_path)
        callback(result, None, save_path=operator.analysis_path)
        assert (Path(operator.analysis_path) / filename).is_file()


def test_spectral_radius_fit_is_deterministic() -> None:
    recurrent = np.diag([0.6, 0.3])
    states = np.zeros((300, 2))
    states[0] = [0.05, -0.04]
    for index in range(states.shape[0] - 1):
        states[index + 1] = np.tanh(states[index] @ recurrent.T)
    first = SpectralRadius(
        leak_rate=1.0, epochs=500, learning_rate=0.02, random_state=3
    )(states)
    second = SpectralRadius(
        leak_rate=1.0, epochs=500, learning_rate=0.02, random_state=3
    )(states)
    np.testing.assert_allclose(first.weights, second.weights)
    np.testing.assert_allclose(first.spectral_radius, 0.6, atol=0.15)
