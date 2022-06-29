import unittest

from miv.signal.generator import generate_random_spiketrain


def test_no_noise():
    spiketrains = generate_random_spiketrain(
        duration=1,
        spikes_per_second=10,
        num_channels=2,
        spike_strength=1,
        noise_multiplier=0,
    )
    for spiketrain in spiketrains:
        assert len(spiketrain) == 10


def test_noise():
    noisy_spiketrains = generate_random_spiketrain(
        duration=1,
        spikes_per_second=10,
        num_channels=2,
        spike_strength=1,
        noise_multiplier=10,
    )
    for spiketrain in noisy_spiketrains:
        assert len(spiketrain) > 10


def test_noise_multiplier():
    spiketrains = generate_random_spiketrain(
        duration=1,
        spikes_per_second=10,
        num_channels=2,
        spike_strength=1,
        noise_multiplier=1,
    )
    for spiketrain in spiketrains:
        assert len(spiketrain) == 20
