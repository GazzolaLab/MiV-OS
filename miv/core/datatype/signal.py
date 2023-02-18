__doc__ = """
Signal
======

.. autoclass:: Signal
   :members:

"""

__all__ = ["Signal"]

from typing import Optional

from dataclasses import dataclass

from miv.core.policy import SupportMultiprocessing
from miv.typing import SignalType, TimestampsType


@dataclass
class Signal(SupportMultiprocessing):
    """
    Contiguous array of raw signal type.

    [signal length, n channels]
    """

    data: SignalType
    timestamps: TimestampsType
    rate: int = 30_000
