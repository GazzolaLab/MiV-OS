__doc__ = """
Basic utilities for serial communication using PySerial package.
The tool is originally developped for experiment using `Stimjim <https://bitbucket.org/natecermak/stimjim/src/master/>`_
The purpose is to output spiketrain as pulse signal.

author: <Seung Hyun Kim skim0119@gmail.com>
"""
__all__ = ["ArduinoSerial", "list_serial_ports"]

import os
import sys
import time

import serial


def list_serial_ports():  # pragma: no cover
    """list serial communication ports available"""
    from serial.tools.list_ports import main

    main()


class ArduinoSerial:
    """
    Stimjim compatible
      - Baudrate: 112500
    """

    def __init__(self, port: str, baudrate: int = 112500):
        self._data_started = False
        self._data_buf = ""
        self._message_complete = False
        self.baudrate = baudrate
        self.port = port
        self.serial_port = None

    def connect(self):
        self.serial_port = self._setup_serial(self.baudrate, self.port)

    def _setup_serial(
        self, baudrate: int, serial_port_name: str, verbose: bool = False
    ):
        """Setup serial connection.

        Parameters
        ----------
        baudrate : int
            serial bits communication rate (bit per sec)
        serial_port_name : str
            Serial port name. Typically start with "COM". To scan available ports,
            run `list_serial_ports`.
        verbose : bool
            If set to true, print out debugging messages (default=False).
        """
        serial_port = serial.Serial(
            port=serial_port_name, baudrate=baudrate, timeout=0, rtscts=True
        )
        if verbose:
            print(f"{serial_port_name=} {baudrate=}")
        self.wait()
        return serial_port

    @property
    def is_open(self):
        return self.serial_port.is_open

    def open(self):
        self.serial_port.open()

    def close(self):
        self.serial_port.close()

    def send(
        self,
        msg: str,
        start_character: str = "",
        eol_character: str = "\n",
        verbose: bool = False,
    ):
        # adds the start- and end-markers before sending
        full_msg = start_character + msg + eol_character
        self.serial_port.write(full_msg.encode("utf-8"))
        if verbose:
            print(f"Msg send: {full_msg}")

    def receive(self, start_character="", eol_character="\n"):
        """receive.

        Parameters
        ----------
        start_character :
            start_character
        eol_character :
            eol_character
        """
        if self.serial_port.in_waiting() > 0 and not self._message_complete:
            x = self.serial_port.read().decode("utf-8")  # decode needed for Python3

            if self._data_started:
                if x != eol_character:
                    self._data_buf = self._data_buf + x
                else:
                    self._data_started = False
                    self._message_complete = True
            elif x == start_character:
                self._data_buf = ""
                self._data_started = True

        if self._message_complete:
            self._message_complete = False
            return self._data_buf
        else:
            return "ready"

    def wait(self, verbose: bool = False):
        """
        Allows time for Arduino launch. It also ensures that any bytes left
        over from a previous message are discarded
        """
        if verbose:
            print("Waiting for Arduino to reset")

        msg = ""
        prev_msg = ""
        while msg.lower().find("ready") == -1:
            msg = self.receive()
            prev_msg = msg
        return prev_msg
