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
from miv.io.serial import StimjimSerial
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
stimjim = StimjimSerial(port="COM3")
# stimjim.connect()
```

## Serial Communication Protocol

The module `ArduionSerial` provides basic operations: `send` and `wait`.
- `send`: Send string message through serial port.
- `wait`: Wait until the buffer is returned. Receive reply from Arduino(Stimjim) device.

```{raw-cell}
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

```{raw-cell}
def write_and_read(s):
    arduino.write(bytes(s, 'utf-8'))
    time.sleep(10)
    data = ser.readlines()
    return data
```

```{raw-cell}
stimjim.send('D\n')
msg = stimjim.wait()
print(msg)
```

```{raw-cell}
cmd = "S0,0,3,100000,1000000;4500,0,50000\n"
```

## Generate Musical Pitch

```{code-cell} ipython3
from math import log2, pow
import IPython
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
import scipy.signal
from miv.coding.temporal import LyonEarModel, BensSpikerAlgorithm
```

```{code-cell} ipython3
A4 = 440
C0 = A4*pow(2, -4.75)
name = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

def pitch(freq):
    h = round(12*log2(freq/C0))
    octave = h // 12
    n = h % 12
    return name[n] + str(octave)
```

### Ben's Spiker Algorithm

```{code-cell} ipython3
frequency = 8
sampling_rate = 40
print(pitch(frequency))
```

```{code-cell} ipython3
t = np.linspace(0,2*np.pi / frequency, int(sampling_rate / (1.0 / (2*np.pi/frequency))))
y = np.sin(frequency * t)
```

```{code-cell} ipython3
bsa = BensSpikerAlgorithm(int(sampling_rate / (1.0 / (2*np.pi/frequency))), threshold=1.1, normalize=True)
spikes, timestamps = bsa(y[:,None])
events = np.where(spikes)[0]
spiketrain = t[events]
```

```{code-cell} ipython3
plt.eventplot(spiketrain)
plt.plot(t,y, 'r')
plt.xlabel('time')
plt.ylabel('y')
plt.show()
```

```{code-cell} ipython3
spiketrain_micro = (spiketrain * 1e6).astype(np.int_)
t_max = int(1e6 * 2 * np.pi / frequency)
```

```{code-cell} ipython3
stimjim.send_spiketrain(0, spiketrain_micro, t_max, 1e6 * 60)
```

```{code-cell} ipython3

```
