__all__ = ["Extendable"]

from typing import List, Protocol, TypeVar


class Extendable(Protocol):
    def extend(self, other) -> None:
        ...


ChannelWiseSelf = TypeVar("ChannelWiseSelf", bound="ChannelWise")


class ChannelWise(Protocol):
    @property
    def number_of_channels(self) -> int:
        ...

    def append(self, other) -> None:
        ...

    def insert(self, index, other) -> None:
        ...

    def __setitem__(self, index, other) -> None:
        ...

    def select(self, indices: List[int]) -> ChannelWiseSelf:
        """
        Select channels by indices.
        """
        ...
