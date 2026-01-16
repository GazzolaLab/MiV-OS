from __future__ import annotations

__doc__ = """
"""
__all__ = [
    "DefaultLoggerMixin",
]

import sys
from typing import TYPE_CHECKING, Any

from loguru import logger

SEPERATOR = " :: "


def configure_logger(start_tag, verbose: int):  # pragma: no cover
    # TODO: Redirect I/O stream to txt
    # Configure logger
    logger.remove()

    mpi_rank_str = ""
    if verbose >= 2:  # Only check for MPI if high verbosity
        try:
            from mpi4py import MPI

            comm = MPI.COMM_WORLD
            rank = comm.Get_rank()
            size = comm.Get_size()
            if size > 1:
                mpi_rank_str = f"<b>[RANK {rank:02d}]</b> "
        except ImportError:
            pass

    # Define formats
    quiet_format = "<level>{message}</level>"
    info_format = "<cyan>{extra[tag]: <20}</cyan> | {message}"
    debug_format = (
        f"{mpi_rank_str}"
        "<green>{time:HH:mm:ss.SSS}</green> | "
        "<level>{level: <8}</level> | "
        "<cyan>{extra[tag]: <20}</cyan> | "  # Padded tag
        "<level>{message}</level>"
    )

    if verbose == 0:
        logger.add(sys.stderr, level="WARNING", format=quiet_format)
    elif verbose == 1:
        # Filter to only show INFO messages from operators
        logger.add(sys.stderr, level="INFO", format=info_format)
    else:  # verbose >= 2
        # Add separate sinks for INFO and DEBUG to use different formats if needed,
        # but here we can use one and rely on the content of the message.
        # Let's keep INFO simple.
        logger.add(
            sys.stderr,
            level="INFO",
            format=debug_format,
            filter=lambda record: record["level"].name == "INFO",
        )
        logger.add(
            sys.stderr,
            level="DEBUG",
            format=debug_format,
            filter=lambda record: record["level"].name == "DEBUG",
        )


class DefaultLoggerMixin:
    tag: str

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        # self.tag is defined in the operator class
        if hasattr(self, "tag"):
            self._logger = logger.bind(tag=self.tag)
        else:
            self._logger = logger.bind(tag=self.__class__.__name__)

    @property
    def logger(self):
        return self._logger
