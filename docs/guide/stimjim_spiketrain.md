---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.13.8
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
---

# Send Spiketrain to Stimjim


Throughout the guide, we use `PySerial` package for serial communication with Stimjim (Arduino base).
This dependency is not included as part of `MiV-OS`, hence user must install it separately:
```bash
pip install pyserial
```

```{code-cell} ipython3
:tags: [hide-cell]

import time
from miv.io.serial import ArduinoSerial
```

## Open Connection

+++

- Baudrate: [here](https://bitbucket.org/natecermak/stimjim/src/c23d98eb90725888241dedc7cab83cacd2bb288e/stimjimPulser/stimjimPulser.ino#lines-246)
- Listing ports: `python -m serial.tools.list_ports`
  - Also you can run as:
  ```py
  from miv.io.serial import list_serial_ports
  list_serial_ports()
  ```

```{code-cell} ipython3
stimjim = ArduinoSerial(port="COM3")
```

## Serial Communication Protocol

The module `ArduionSerial` provides basic operations: `send` and `wait`.
- `send`: Send string message through serial port.
- `wait`: Wait until the buffer is returned. Receive reply from Arduino(Stimjim) device.

```{code-cell} ipython3
count = 0
max_iteration = 100
prevTime = time.time()
for i in range(max_iteration):
    # check for a reply from board
    reply = ""
    stimjim.send(f"Test Message: {count}")
    msg = stimjim.wait()
    print (f"Time {time.time()}")
    print(msg)


    prevTime = time.time()
    count += 1
```

> Check if `.is_open` returns `True` before running complex stimulation.
> If it refuse to open communication port due to `Permission` issue, check if `Serial Monitor` is opened on `Arduino IDE` (Or check for any other serial connection).

+++

## Example Stimulation Run

```{code-cell} ipython3
def write_and_read(s):
    arduino.write(bytes(s, 'utf-8'))
    time.sleep(10)
    data = ser.readlines()
    return data
```

```{code-cell} ipython3
stimjim.send('D\n')
msg = stimjim.wait()
print(msg)
```

```{code-cell} ipython3
cmd = "S0,0,3,100000,1000000;4500,0,50000\n"
```
