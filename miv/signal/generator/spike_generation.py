__all__ = ["generate_random_spiketrain"]

from typing import List

import numpy as np
from neo.core import SpikeTrain
from quantities import s

from miv.typing import SpikestampsType


def generate_random_spiketrain(
    duration: float,
    spikes_per_second: float,
    num_channels: int,
    random_spike_strength: bool = False,
    spike_strength: int = 3,
    random_noise_multiplier: bool = False,
    noise_multiplier: float = 10,
) -> List[SpikestampsType]:
    """Generate spiketrains with noise

    Parameters
    ----------
    duration : float
        Length of recording in seconds
    spikes_per_second : float
        Number of spikes per second
    num_channels : int
        Number of channels
    random_spike_strength : bool
        Generates random spike strength values for each channel
    spike_strength : int
        The number of times each spike will be repeated
    random_noise_multiplier : bool
        Generates random noise multipliers for each channel
    noise_multiplier : float
        The amount of noise spikes. This value is the ratio of noise spikes to real spikes.

    Returns
    -------
    artifical_spikes : List[SpikestampsType]

    """

    artificial_spikes = []

    for chan in range(num_channels):
        # Step one: generate true spikes
        if random_spike_strength:
            spike_strength = int(np.random.rand() * 10)
        channel_spikes = np.repeat(
            np.arange(start=0, stop=duration, step=1 / spikes_per_second),
            spike_strength,
        )
        # Step two: add noise
        if random_noise_multiplier:
            noise_multiplier = 10 ** (np.random.rand() * 2)
        noise_count = int(spikes_per_second * duration * noise_multiplier)
        channel_spikes = np.concatenate(
            (channel_spikes, np.random.rand(noise_count) * duration)
        )

        artificial_spikes.append(SpikeTrain(channel_spikes * s, duration))

    return artificial_spikes
