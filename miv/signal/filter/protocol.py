__doc__ = ""
__all__ = ["FilterProtocol"]

from typing import Protocol


from miv.core.datatype import Signal


class FilterProtocol(Protocol):
    """Behavior definition of all filter operator."""

    tag: str = ""

    def __call__(self, array: Signal) -> Signal:
        """User can apply the filter by directly calling.
        Parameters
        ----------
        array : Signal
        """
        ...

    def __repr__(self) -> str:
        """String representation for interactive debugging."""
        ...
