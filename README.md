<div align='center'>
<h1> MiV-OS: Spike Analysis and Computing Framework </h1>

[![License][badge-LICENSE]][link-LICENSE]
[![Release pypi][badge-pypi]][link-pypi]
[![Build Status][badge-CI]][link-CI]
[![Documentation Status][badge-docs-status]][link-docs-status]
[![Downloads][badge-pepy-download-count]][link-pepy-download-count]
[![codecov][badge-codecov]][link-codecov]

</div>

---

Python analysis and computing framework developed for [Mind-in-Vitro(MiV)][link-project-website] project.

## Installation
[![PyPI version][badge-pypi]][link-pypi]

MiV-OS is compatible with Python 3.10+. The easiest way to install is using python installation package (PIP)

~~~bash
$ pip install MiV-OS
~~~

## Documentation
[![Documentation Status][badge-docs-status]][link-docs-status]

Documentation of the package is available [here][link-docs-status]

## Published demo

The RC–KT publication analysis is expressed as a branched MiV pipeline:

```python
from miv.core import Pipeline
from miv.io.file import ImportSignal
from miv.signal import ButterBandpass, ThresholdCutoff
from miv.statistics import BayesianAdaptiveKernelSmoother
from miv.statistics.connectivity import DirectedConnectivity
from miv.statistics.criticality import BranchingRatio
from miv.statistics.reservoir import (
    ExponentialSpikeEncoder,
    KernelRank,
    RidgeReadout,
    SpectralRadius,
    StimulusTrializer,
)

ephys = ImportSignal("recording.h5", group="Ephys")
stimulus = ImportSignal("recording.h5", group="Stimulus")
bandpass = ButterBandpass(lowcut=400, highcut=1500)
spikes = ThresholdCutoff(cutoff=5.0)
trials = StimulusTrializer()
encoding = ExponentialSpikeEncoder(decay_rate=5.0)
readout = RidgeReadout(random_state=0)

ephys >> bandpass >> spikes
spikes >> trials
stimulus >> trials
trials >> encoding >> readout

baks = BayesianAdaptiveKernelSmoother()
branching = BranchingRatio()
connectivity = DirectedConnectivity()
rank = KernelRank()
radius = SpectralRadius(random_state=0)

spikes >> baks
spikes >> branching
spikes >> connectivity
encoding >> rank
encoding >> radius

Pipeline([readout, baks, branching, connectivity, rank, radius]).run(
    working_directory="results/rc_kt"
)
```

The code is available on the [`pub/RC-KT` publication branch][link-rc-kt-demo].
Its integration PR will be merged into `main`, and the accompanying dataset
released, upon publication.

## Contribution

If you would like to participate, please read our [contribution guideline](CONTRIBUTING.md)

The development of MiV-OS is lead by the [Gazzola Lab][link-lab-website] at the University of Illinois at Urbana-Champaign.

## List of publications and submissions

## Citation

```
@misc{MiV-OS,
  author = {Gazzola Lab},
  title = {MiV-OS: Analysis and Computation Framework on MiV System and Simulator},
  year = {2022},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/GazzolaLab/MiV-OS}},
}
```

```
@misc{Kim2026RC-KT,
    title={Computing with Living Neurons: Chaos-Controlled Reservoir Computing with Knowledge Transplant},
    author={Seung Hyun Kim and Zhi Dou and Gaurav Upadhyay and Anay Pattanaik and Leo Maslov and Lav Varshney and John Beggs and Howard Gritton and Mattia Gazzola},
    year={2026},
    eprint={2604.02552},
    archivePrefix={arXiv},
    primaryClass={cs.NE},
    url={https://arxiv.org/abs/2604.02552},
}
```

We ask that any publications which use MiV-OS package to cite the above papers.

## Developers ✨
_Names arranged alphabetically_
- Arman Tekinalp
- Andrew Dou
- [Frithjof Gressmann](https://github.com/frthjf)
- Gaurav Upadhyay
- [Seung Hyun Kim](https://github.com/skim0119)

[//]: # (Collection of URLs.)

[link-lab-website]: http://mattia-lab.com/
[link-project-website]: https://mindinvitro.illinois.edu/
[link-docs-status]: https://miv-os.readthedocs.io/en/latest/?badge=latest
[link-CI]: https://github.com/GazzolaLab/MiV-OS/actions
[link-LICENSE]: https://opensource.org/licenses/MIT
[link-pypi]: https://badge.fury.io/py/MiV-OS
[link-pepy-download-count]: https://pepy.tech/project/MiV-OS
[link-codecov]: https://codecov.io/gh/GazzolaLab/MiV-OS
[link-rc-kt-demo]: https://github.com/GazzolaLab/MiV-OS/tree/pub/RC-KT/examples/rc_kt

[//]: # (Collection of Badges)

[badge-docs-status]: https://readthedocs.org/projects/miv-os/badge/?version=latest
[badge-CI]: https://github.com/GazzolaLab/MiV-OS/workflows/CI/badge.svg
[badge-LICENSE]: https://img.shields.io/badge/License-MIT-yellow.svg
[badge-pypi]: https://badge.fury.io/py/MiV-OS.svg
[badge-pepy-download-count]: https://static.pepy.tech/badge/MiV-OS
[badge-codecov]: https://codecov.io/gh/GazzolaLab/MiV-OS/branch/main/graph/badge.svg?token=OM5LYWF5KP
