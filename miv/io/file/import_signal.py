__all__ = ["ImportSignal"]

from collections.abc import Generator


from miv.core import Signal
from miv.core.source.node_mixin import DataLoaderMixin
from miv.io import file as miv_file


class ImportSignal(DataLoaderMixin):
    def __init__(
        self,
        data_path: str,
        group: str = "Ephys",
        subset: int | list[int] | tuple[int, int] | None = None,
        tag: str = "import signal",
    ) -> None:
        self.data_path: str = data_path
        self.group = group
        self.subset = subset
        self.tag: str = f"{tag}"
        super().__init__()

    def load(self) -> Generator[Signal]:
        data, container = miv_file.read(
            self.data_path, groups=self.group, subset=self.subset
        )
        num_container = data["_NUMBER_OF_CONTAINERS_"]
        self.logger.info(f"Loading: {num_container=}")

        for i in range(num_container):
            miv_file.unpack(container, data, i)
            signal = Signal(
                data=container[f"{self.group}/Data"],
                timestamps=container[f"{self.group}/Timestamps"],
                rate=container[f"{self.group}/Rate"],
            )
            self.logger.info(
                f"{i}-container | {signal.data.shape=}, {signal.timestamps.shape=}, {signal.rate=}"
            )
            yield signal
