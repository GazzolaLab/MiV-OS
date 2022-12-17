from unittest.mock import patch

import numpy as np
import pytest

from miv.io.serial import ArduinoSerial, list_serial_ports


def test_list_serial_ports():
    # Test that the list_serial_ports function calls the main function of the
    # serial.tools.list_ports module
    list_serial_ports()
    assert 1


def test_arduino_module_init():
    # Test that the __init__ method correctly initializes the object's attributes
    port = "/dev/ttyACM0"
    baudrate = 112500
    arduino_serial = ArduinoSerial(port=port, baudrate=baudrate)
    assert arduino_serial.port == port
    assert arduino_serial.baudrate == baudrate
    assert arduino_serial._data_started is False
    assert arduino_serial._data_buf == ""
    assert arduino_serial._message_complete is False
    assert arduino_serial.serial_port is None
