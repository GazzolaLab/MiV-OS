from typing import Protocol
from collections.abc import Iterable
from functools import reduce


class _Collapsable(Protocol):
    """A protocol defining the concatenate method for collapsable objects."""

    @classmethod
    def concatenate(
        cls, values: Iterable["_Collapsable"], *, head: "_Collapsable | None" = None
    ) -> "_Collapsable": ...

    def extend(self, other: "_Collapsable") -> "_Collapsable": ...


class ConcatenateMixin:
    """A mixin providing concatenate functionality for collapsable objects.
    Allows combining multiple objects by extending them sequentially."""

    @classmethod
    def concatenate(
        cls, values: Iterable[_Collapsable], *, head: "_Collapsable | None" = None
    ) -> _Collapsable:
        """
        Concatenate a list of collapsable objects.
        Note, the method change the data in-place. The concatenated result will be
        stacked on the first object. if `head` is provided, the concatenated result will be
        stacked on the `head`.

        Args:
            values: A list of collapsable objects to concatenate.
            head: An optional collapsable object to use as the head of the concatenation.

        Returns:
            A new collapsable object that is the result of concatenating the input objects.
            The original objects are not modified.

        """
        if head is None:
            return reduce(lambda x, y: x.extend(y), values)
        else:
            return reduce(lambda x, y: x.extend(y), values, head)

    # Deprecated alias
    from_collapse = concatenate
