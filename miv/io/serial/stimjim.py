__doc__ = """
Stimjim extension.
Basic utilities for translating `spiketrains` into Stimjim pulse generator.

author: <Seung Hyun Kim skim0119@gmail.com>
"""
__all__ = ["StimjimSerial"]

from typing import Any


import numpy as np

from miv.io.serial import ArduinoSerial


class StimjimSerial(ArduinoSerial):
    """
    Module to control Stimjim using PySerial.

    All time-units are in micro.
    All volt-units are in milli.
    All ampere-units are in micro.
    """

    def __init__(
        self,
        port: str,
        output0_mode: int = 1,
        output1_mode: int = 3,
        high_v: int = 4500,
        low_v: int = 0,
        **kwargs: Any,
    ) -> None:
        super().__init__(port, **kwargs)
        self.output0_mode = output0_mode
        self.output1_mode = output1_mode
        self.high_v_1 = high_v
        self.low_v_1 = low_v
        self.high_v_2 = 0
        self.low_v_2 = 0

    def send_spiketrain(
        self,
        pulsetrain: int,
        spiketrain: np.ndarray,
        t_max: int,
        total_duration: int,
        delay: float = 0.0,
        channel: int = 0,
        reverse: bool = False,
    ) -> str:
        total_string, total_period = self._spiketrain_to_str(
            spiketrain, t_max, reverse=reverse
        )
        total_string.insert(
            0,
            self._start_str(
                pulsetrain,
                self.output0_mode,
                self.output1_mode,
                total_period,
                total_duration,
            ),
        )
        return "; ".join(total_string)

    def _start_str(
        self,
        pulsetrain: int,
        output0_mode: int,
        output1_mode: int,
        period: int,
        duration: int,
    ) -> str:
        return f"S{pulsetrain},{output0_mode},{output1_mode},{period},{duration}"

    def _spiketrain_to_str(
        self,
        spiketrain: np.ndarray,
        t_max: int,
        pulse_length: int = 10_000,
        reverse: bool = False,
    ) -> tuple[list[str], int]:
        """
        Convert a spiketrain into a series of strings that can be sent to the Stimjim device.
        """
        spiketrain = np.insert(spiketrain, 0, 0)
        spiketrain = np.append(spiketrain, t_max)
        gaps = np.diff(spiketrain).astype(np.int_)
        if reverse:
            gaps = gaps[::-1]
        if np.any(gaps < pulse_length):
            raise ValueError(
                f"Gap between pulse {gaps} must be larger than pulse length {pulse_length}. {spiketrain}"
            )

        # String functions
        def gap_to_str(x: int, A1: int, A2: int) -> str:
            return f"{A1},{A2},{x:d}"

        pulse_to_str = gap_to_str(pulse_length, self.high_v_1, 0)

        total_string = [pulse_to_str, gap_to_str(gaps[0], self.low_v_1, 0)]  # First Gap
        for gap in gaps[1:]:
            total_string.append(pulse_to_str)
            total_string.append(gap_to_str(gap - pulse_length, self.low_v_1, 0))

        total_period = gaps.sum()
        return total_string, total_period
