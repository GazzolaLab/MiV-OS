<div align='center'>
<h1> MiV-OS: Composable Data Analysis Pipelines </h1>

[![License][badge-LICENSE]][link-LICENSE]
[![Release pypi][badge-pypi]][link-pypi]
[![Build Status][badge-CI]][link-CI]
[![Documentation Status][badge-docs-status]][link-docs-status]
[![Downloads][badge-pepy-download-count]][link-pepy-download-count]
[![codecov][badge-codecov]][link-codecov]

</div>

---

MiV-OS is a Python framework for turning data analysis into an explicit,
reusable pipeline. It began in the [Mind in Vitro][link-project-website]
project, where electrophysiology recordings must move through many connected
steps, but its core model is not tied to one experiment or data type.

Instead of making a script responsible for both *what the analysis means* and
*how every step should run*, MiV-OS represents an analysis as a graph of small
operators. The graph can be inspected, branched into several results, cached,
and adapted as the data or computing environment changes.

- **Compose** loaders and transformations into a readable analysis graph.
- **Reuse** a scientific step in another pipeline without copying its control flow.
- **Scale deliberately** with caching, streaming operators, and alternative runners.
- **Keep results explainable** by preserving the path from source data to outcome.

Electrophysiology is the package's first and richest collection of tools—not a
requirement for using the framework. Start with [About MiV-OS][link-about-miv-os]
for the design philosophy, or go directly to the [core
quickstart][link-core-quickstart] to build a small pipeline.

## Installation
[![PyPI version][badge-pypi]][link-pypi]

MiV-OS is compatible with Python 3.10+. The easiest way to install is using python installation package (PIP)

~~~bash
$ pip install MiV-OS
~~~

## Documentation
[![Documentation Status][badge-docs-status]][link-docs-status]

The [documentation][link-docs-status] is arranged as a gradual tour: begin
with the purpose and core model, build a first graph, then explore practical
guides, electrophysiology tools, advanced execution, and the API reference.

## Published demo

The analysis pipeline and demonstration of RC–KT results are built upon MiV-OS
pipelines so they can be reproduced, reused, and adapted to other datasets. The
imports are shared by the three analysis graphs below:

```python
from miv.core import Pipeline
from miv.io.file import ImportSignal
from miv.signal import ButterBandpass, ThresholdCutoff
from miv.statistics import (
    BayesianAdaptiveKernelSmoother,
    ExponentialSpikeEncoder,
    FixedDurationTrializer,
    KernelRank,
    SpectralRadius,
)
from miv.statistics.connectivity import DirectedConnectivity
from miv.statistics.criticality import BranchingRatio
```

### Statistical diagnostics

```python
ephys = ImportSignal("recording.h5", group="Ephys")
bandpass = ButterBandpass(lowcut=400, highcut=1500)
spikes = ThresholdCutoff(cutoff=5.0)

ephys >> bandpass >> spikes

baks = BayesianAdaptiveKernelSmoother(sample_rate=100.0)
branching = BranchingRatio()
connectivity = DirectedConnectivity(seed=0)
diagnostic_trials = FixedDurationTrializer(trial_duration=1.0)
diagnostic_states = ExponentialSpikeEncoder(decay_rate=5.0)
rank = KernelRank()
radius = SpectralRadius(random_state=0)

spikes >> baks
spikes >> branching
spikes >> connectivity
spikes >> diagnostic_trials >> diagnostic_states
diagnostic_states >> rank
diagnostic_states >> radius

Pipeline([baks, branching, connectivity, rank, radius]).run(
    working_directory="results/statistical_diagnostics"
)
```

The code is available on the [`pub/RC-KT` publication branch][link-rc-kt-demo]
and tracked in [draft integration PR #589][link-rc-kt-pr]. The PR will be
merged into `main`, and the accompanying dataset released, upon publication.

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
[link-about-miv-os]: https://miv-os.readthedocs.io/en/latest/about.html
[link-core-quickstart]: https://miv-os.readthedocs.io/en/latest/tutorial/core_quickstart.html
[link-CI]: https://github.com/GazzolaLab/MiV-OS/actions
[link-LICENSE]: https://opensource.org/licenses/MIT
[link-pypi]: https://badge.fury.io/py/MiV-OS
[link-pepy-download-count]: https://pepy.tech/project/MiV-OS
[link-codecov]: https://codecov.io/gh/GazzolaLab/MiV-OS
[link-rc-kt-demo]: https://github.com/GazzolaLab/MiV-OS/tree/pub/RC-KT/examples/rc_kt
[link-rc-kt-pr]: https://github.com/GazzolaLab/MiV-OS/pull/589

[//]: # (Collection of Badges)

[badge-docs-status]: https://readthedocs.org/projects/miv-os/badge/?version=latest
[badge-CI]: https://github.com/GazzolaLab/MiV-OS/workflows/CI/badge.svg
[badge-LICENSE]: https://img.shields.io/badge/License-MIT-yellow.svg
[badge-pypi]: https://badge.fury.io/py/MiV-OS.svg
[badge-pepy-download-count]: https://static.pepy.tech/badge/MiV-OS
[badge-codecov]: https://codecov.io/gh/GazzolaLab/MiV-OS/branch/main/graph/badge.svg?token=OM5LYWF5KP
