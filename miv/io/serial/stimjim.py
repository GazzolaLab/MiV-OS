__doc__ = """
Stimjim extension.
Basic utilities for translating `spiketrains` into Stimjim pulse generator.

author: <Seung Hyun Kim skim0119@gmail.com>
"""
__all__ = ["StimjimSerial"]

from typing import List, Optional

import os
import sys
import time

import numpy as np
import serial

from miv.io.serial import ArduinoSerial
from miv.typing import SpiketrainType


class StimjimSerial(ArduinoSerial):
    """
    Module to control Stimjim using PySerial.

    All time-units are in micro.
    All volt-units are in milli.
    All ampere-units are in micro.
    """

    def __init__(
        self, port, output0_mode=1, output1_mode=3, high_v=4500, low_v=0, **kwargs
    ):
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
        spiketrain: SpiketrainType,
        t_max: int,
        total_duration: int,
        delay: float = 0.0,
        channel: int = 0,
        reverse: bool = False,
    ) -> bool:
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

    def _start_str(self, pulsetrain, output0_mode, output1_mode, period, duration):
        return f"S{pulsetrain},{output0_mode},{output1_mode},{period},{duration}"

    def _spiketrain_to_str(
        self,
        spiketrain: SpiketrainType,
        t_max: int,
        pulse_length: int = 10_000,
        reverse: bool = False,
    ) -> List[str]:
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
        def gap_to_str(x, A1, A2):
            return f"{A1},{A2},{x:d}"

        pulse_to_str = gap_to_str(pulse_length, self.high_v_1, 0)

        total_string = [pulse_to_str, gap_to_str(gaps[0], self.low_v_1, 0)]  # First Gap
        for gap in gaps[1:]:
            total_string.append(pulse_to_str)
            total_string.append(gap_to_str(gap - pulse_length, self.low_v_1, 0))

        total_period = gaps.sum()
        return total_string, total_period
