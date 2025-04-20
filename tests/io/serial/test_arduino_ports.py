# from unittest.mock import patch
import pytest
import numpy as np


def test_arduino_module_init():
    # Test that the __init__ method correctly initializes the object's attributes
    pytest.importorskip("serial")
    from miv.io.serial import ArduinoSerial

    port = "/dev/ttyACM0"
    baudrate = 112500
    arduino_serial = ArduinoSerial(port=port, baudrate=baudrate)
    assert arduino_serial.port == port
    assert arduino_serial.baudrate == baudrate
    assert arduino_serial._data_started is False
    assert arduino_serial._data_buf == ""
    assert arduino_serial._message_complete is False
    assert arduino_serial.serial_port is None
