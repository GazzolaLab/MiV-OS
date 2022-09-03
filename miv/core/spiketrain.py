__all__ = ["SpikeTrain"]

import neo


class SpikeTrain(neo.core.SpikeTrain):
    """Array of spike times

    Represents spikes emitted by the same unit in a period of times.
    """

    # this is currently identical to neo.core.SpikeTrain
    # but may deviate in future
