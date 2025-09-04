__doc__ = """Concatenate operation for collapsable objects."""
__all__ = ["concatenate"]

from typing import TYPE_CHECKING, Protocol
from functools import reduce


class _Collapsable(Protocol):
    """A protocol defining the concatenate method for collapsable objects."""

    def extend(self, other: "_Collapsable") -> "_Collapsable":
        """
        Note::
            The extend method should be compatible against empty objects.
            i.e.
                cls.empty().extend([obj]) == obj
        """

    @classmethod
    def empty(cls) -> "_Collapsable":
        """
        This class should not take any arguments.
        Datatype must support purely empty objects.
        """


def concatenate(
    values: list[_Collapsable], *, head: _Collapsable | None = None
) -> _Collapsable:
    if len(values) == 0:
        raise RuntimeError("Cannot concatenate empty list")
    if head is None:
        head = type(values[0]).empty()
    reduce(lambda acc, val: (acc.extend(val) or acc), values, head)
    # Equivalent to::
    # for value in values:
    #     out.extend(value)
    return head


if TYPE_CHECKING:
    from miv.core.datatype.signal import Signal
    from miv.core.datatype.events import Events
    from miv.core.datatype.spikestamps import Spikestamps

    _: list[Signal | Spikestamps] = [
        Signal.empty(),
        Spikestamps.empty(),
        Events.empty(),
    ]
