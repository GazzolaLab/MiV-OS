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
    GPFALatentProjector,
    KernelRank,
    KnowledgeTransfer,
    KnowledgeTransferInputBuilder,
    KnowledgeTransferTrialSelector,
    RidgeReadout,
    SpectralRadius,
    TTLPulseDecoder,
)
from miv.statistics.connectivity import DirectedConnectivity
from miv.statistics.criticality import BranchingRatio
from miv_state_space import HierarchicalGPFA
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

### Reservoir computing

```python
stimulus = ImportSignal("recording.h5", group="Stimulus")
decoder = TTLPulseDecoder(stimulus_channel=0)
trials = FixedDurationTrializer(trial_duration=1.0)
encoding = ExponentialSpikeEncoder(decay_rate=5.0)
readout = RidgeReadout(random_state=0)

stimulus >> decoder
spikes >> trials
decoder >> trials
trials >> encoding >> readout

Pipeline(readout).run(working_directory="results/rc")
```

### Knowledge-transplanted reservoir computing

```python
# ``expert_trials`` and ``student_trials`` are the paired outputs of the
# corresponding import, filtering, detection, and trialization graphs shown
# in examples/rc_kt/KT.py.
expert_gpfa = HierarchicalGPFA(random_state=0)
student_gpfa = HierarchicalGPFA(random_state=0)
kt_input = KnowledgeTransferInputBuilder()
expert_trial_stream = KnowledgeTransferTrialSelector("expert")
student_trial_stream = KnowledgeTransferTrialSelector("student")
expert_features = GPFALatentProjector(9, "expert")
student_features = GPFALatentProjector(9, "student")
expert_readout = RidgeReadout(random_state=0)
transplant = KnowledgeTransfer(band_dimensions=(3, 3, 3))

expert_trials >> kt_input
student_trials >> kt_input
kt_input >> expert_trial_stream >> expert_gpfa
kt_input >> student_trial_stream >> student_gpfa
expert_gpfa >> student_gpfa  # serial dependency: freeze expert kernels
expert_gpfa >> expert_features
kt_input >> expert_features
student_gpfa >> student_features
kt_input >> student_features
expert_features >> expert_readout
expert_features >> transplant
student_features >> transplant
expert_readout >> transplant

Pipeline(transplant).run(working_directory="results/kt_rc")

kt_result = transplant.output()
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
