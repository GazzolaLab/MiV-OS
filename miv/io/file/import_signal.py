__all__ = ["ImportSignal"]

from collections.abc import Generator


from miv.core.datatype import Signal
from miv.core.operator.operator import DataLoaderMixin
from miv.io import file as miv_file


class ImportSignal(DataLoaderMixin):
    def __init__(
        self,
        data_path: str,
        tag: str = "import signal",
    ) -> None:
        self.data_path: str = data_path
        super().__init__()
        self.tag: str = f"{tag}"

    def load(self) -> Generator[Signal]:
        data, container = miv_file.read(self.data_path)
        num_container = data["_NUMBER_OF_CONTAINERS_"]
        self.logger.info(f"Loading: {num_container=}")

        for i in range(num_container):
            miv_file.unpack(container, data, i)
            signal = Signal(
                data=container["Ephys/Data"],
                timestamps=container["Ephys/Timestamps"],
                rate=container["Ephys/Rate"],
            )
            self.logger.info(
                f"{i}-container | {signal.data.shape=}, {signal.timestamps.shape=}, {signal.rate=}"
            )
            yield signal
