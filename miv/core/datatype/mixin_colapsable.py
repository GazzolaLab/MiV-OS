from typing import Protocol
from collections.abc import Iterable


class _Collapsable(Protocol):
    """A protocol defining the concatenate method for collapsable objects."""

    @classmethod
    def concatenate(
        cls, values: Iterable["_Collapsable"], *, head: "_Collapsable | None" = None
    ) -> "_Collapsable": ...

    def extend(self, other: "_Collapsable") -> "_Collapsable": ...

    @classmethod
    def empty(cls, *args) -> "_Collapsable": ...


class ConcatenateMixin:
    """A mixin providing concatenate functionality for collapsable objects.
    Allows combining multiple objects by extending them sequentially."""

    @classmethod
    def concatenate(
        cls,
        values: list[_Collapsable],
    ) -> _Collapsable:
        """
        Concatenate a list of collapsable objects.
        Note, the method change the data in-place. The concatenated result will be
        stacked on the first object.

        Args:
            values: A list of collapsable objects to concatenate.

        Returns:
            A new collapsable object that is the result of concatenating the input objects.
            The original objects are not modified.

        """
        volume = values[0].empty(values[0].number_of_channels)
        for value in values[1:]:
            # TODO: assert compatibility
            volume.extend(value)
        return volume
