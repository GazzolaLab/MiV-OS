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

# Temporal Encoding: Lyon Ear Model

- Schroeder, M. R. (1973). An integrable model for the basilar membrane. The Journal of the Acoustical Society of America, 53(2), 429–434. doi:10.1121/1.1913339
- Zweig, G., Lipes, R., & Pierce, J. R. (1976). The cochlear compromise. Journal of the Acoustical Society of America, 59(4), 975–982. doi:10.1121/1.380956
- Lyon, R. F. (1982). A Computational Model of Filtering, Detection, and Compreion in the Cochlea. IEEE Artificial IntelligenceArtificial Intelligence, 3–6.
- Lyon, R. F., & Lauritzen, N. (1985). Processing Speech With the Multi-Serial Signal Processor. ICASSP, IEEE International Conference on Acoustics, Speech and Signal Processing - Proceedings, 981–984. doi:10.1109/icassp.1985.1168158
- Slaney, M. (1988). Lyon’s Cochlear Model. 1–79.
- Van, L. M. (1992). Pitch and voiced/unvoiced determination with an auditory model. Journal of the Acoustical Society of America, 91(6), 3511–3526. doi:10.1121/1.402840
- Gabbiani, Fabrizio. (1996). Coding of time-varying signals in spike trains of linear and half-wave rectifying neurons. Network: Computation in Neural Systems, 7(1), 61–85. doi:10.1080/0954898x.1996.11978655
- Reich, D. S., Victor, J. D., & Knight, B. W. (1998). The power ratio and the interval map: Spiking models and extracellular recordings. Journal of Neuroscience, 18(23), 10090–10104. doi:10.1523/jneurosci.18-23-10090.1998
- Gabbiani, F., & Metzner, W. (1999). Encoding and processing of sensory information in neuronal spike trains. Journal of Experimental Biology, 202(10), 1267–1279. doi:10.1242/jeb.202.10.1267
- Schrauwen, B., & Van Campenhout, J. (2003). BSA, a Fast and Accurate Spike Train Encoding Scheme. Proceedings of the International Joint Conference on Neural Networks, 4, 2825–2830. doi:10.1109/ijcnn.2003.1224019
- Cosi, P., & Zovato, E. (2012). Lyon ’ s Auditory Model Inversion : a Tool for Sound Separation and Speech Enhancement. 3–6.
- Swanson, B. A. (2015). Pitch Perception with Cochlear Implants. (August 2008).
- Zai, A. T., Bhargava, S., Mesgarani, N., & Liu, S. C. (2015). Reconstruction of audio waveforms from spike trains of artificial cochlea models. Frontiers in Neuroscience, 9(OCT), 1–13. doi:10.3389/fnins.2015.00347
- Jin, Y., & Li, P. (2017). Performance and robustness of bio-inspired digital liquid state machines: A case study of speech recognition. Neurocomputing, 226(September 2015), 145–160. doi:10.1016/j.neucom.2016.11.045
- Huang, N., Slaney, M., & Elhilali, M. (2018). Connecting deep neural networks to physical, perceptual, and electrophysiological auditory signals. Frontiers in Neuroscience, 12(AUG), 1–14. doi:10.3389/fnins.2018.00532
- Petro, B., Kasabov, N., & Kiss, R. M. (2020). Selection and Optimization of Temporal Spike Encoding Methods for Spiking Neural Networks. IEEE Transactions on Neural Networks and Learning Systems, 31(2), 358–370. doi:10.1109/TNNLS.2019.2906158

- Xu, Y., Thakur, C. S., Singh, R. K., Hamilton, T. J., Wang, R. M., & van Schaik, A. (2018). A FPGA implementation of the CAR-FAC cochlear model. Frontiers in Neuroscience, 12(APR), 1–14. doi:10.3389/fnins.2018.00198

+++

## Quick Demo

```{code-cell} ipython3
:tags: [hide-cell]

import IPython
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
```

```{code-cell} ipython3
from miv.converter.temporal import LyonEarModel, BensSpikerAlgorithm
```

```{code-cell} ipython3
filepath = "Cochlear-Implant-Processor/Samples/Birds.wav"
IPython.display.Audio(filepath)
```

```{code-cell} ipython3
sampling_rate, waveform = wavfile.read(filepath)
waveform = waveform.astype(np.float_)
print(f"{sampling_rate=}, {waveform.shape=}")

left_wave = np.ascontiguousarray(waveform[:,0])
right_wave = np.ascontiguousarray(waveform[:,1])
```

### Lyon Ear Model

```{code-cell} ipython3
decimation_factor = 64
ear_model = LyonEarModel(sampling_rate=sampling_rate, decimation_factor=decimation_factor)
```

```{code-cell} ipython3
decimation_factor = 64
cochlear_left = ear_model(left_wave)
cochlear_right = ear_model(right_wave)
print(cochlear_right.shape)
```

```{code-cell} ipython3
fig = plt.figure(figsize=(16,8))
ax1 = fig.add_subplot(211)
img1 = ax1.imshow(cochlear_right.T)
ax1.set_aspect('auto')
plt.colorbar(img1)
ax2 = fig.add_subplot(212)
img2 = ax2.imshow(cochlear_right.T)
ax2.set_aspect('auto')
plt.colorbar(img2)
```

## BSA

```{code-cell} ipython3
out = np.stack([cochlear_left, cochlear_right])
```

```{code-cell} ipython3
bsa = BensSpikerAlgorithm(out)
spiketrain = bsa.get_spikes()[0]
```

```{code-cell} ipython3
# Binning
spiketime = [[] for _ in range(spiketrain.shape[1])]
spikestamp = np.where(spiketrain)
for i, j in zip(spikestamp[0], spikestamp[1]):
    spiketime[j].append(i)
```

### Replay

```{code-cell} ipython3
from matplotlib import animation
plt.rcParams["animation.html"] = "jshtml"
plt.rcParams["figure.dpi"] = 150
plt.ioff()
```

```{code-cell} ipython3
fig, ax = plt.subplots()

window = 0.15
tfinal = 10.0
dps = int(spiketrain.shape[0] / tfinal)
fps = 60.0
def animate(frame):
    t = frame / fps
    plt.cla()
    low = t * dps
    high = (t + window) * dps
    plt.eventplot([np.array(list(filter(lambda x: x >= low and x <= high, sp)))/dps for sp in spiketime])
    plt.xlim(t,t+window)
    plt.xlabel('time (sec)')
    plt.ylabel('channel')
    plt.title(f'{t=:.02f} {frame=}')
    ax.set_aspect('auto')
animation.FuncAnimation(fig, animate, frames=int(tfinal * fps))
```

## Testing (Sinusoidal)

```{code-cell} ipython3
raise Exception # Do not proceed beyond
```

```{code-cell} ipython3
out = calc.lyon_passive_ear(data[:,0], samplerate)
print('Out:')
print(out.shape)
```

```{code-cell} ipython3
data2 = np.sin(np.arange(2041)/20000*2*np.pi*1000)
out = calc.lyon_passive_ear(data2, 20000, 20)
```

```{code-cell} ipython3
plt.imshow(out, aspect='auto')
plt.colorbar()
```

```{code-cell} ipython3
data3 = np.zeros(256); data3[0] = 1
out = calc.lyon_passive_ear(data3, 16000, 1)
out = np.fmin(out, 0.0004)
```

```{code-cell} ipython3
plt.imshow(out, aspect='auto')
plt.colorbar()
```

```{code-cell} ipython3

```
