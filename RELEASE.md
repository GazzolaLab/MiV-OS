# Release Note (v0.3.0-beta.0)

## What's Changed

* Major rework in core operators and datatypes @skim0119 in https://github.com/GazzolaLab/MiV-OS/pull/204
  - Includes pipeline, lazy generator runs, caching, and callback
* Underlying structure implementation is done. The remaining work is left to test and try on various experimental procedures.
* Reduced documentation to only includes the pages that support the current operator structure.

**Full Changelog**: https://github.com/GazzolaLab/MiV-OS/commits/v0.3.0-beta.0


# Release Note (version 0.2.4)

## What's Changed

### Features

* Dynamic Time Warping for sorting and clustering by @skim0119 in https://github.com/GazzolaLab/MiV-OS/pull/181
* Make `load_fragments` to be default behavior for `load` by @eunice-chan in https://github.com/GazzolaLab/MiV-OS/pull/190
* Add more features for `miv.core.Spikestamps`: time-window view, first/last spikestamps, and spike counter

### Documentation

* update raw-binary file pre-processing by @skim0119 in https://github.com/GazzolaLab/MiV-OS/pull/191
* Update read_TTL_events guide by @skim0119 in https://github.com/GazzolaLab/MiV-OS/pull/187
* how to use change-point detection algorithm by @skim0119 in https://github.com/GazzolaLab/MiV-OS/pull/193

## New Contributors

* @eunice-chan made their first contribution in https://github.com/GazzolaLab/MiV-OS/pull/190

**Full Changelog**: https://github.com/GazzolaLab/MiV-OS/compare/v0.2.3...V0.2.4

# Release Note (version 0.2.3)

## What's Changed

### Features

* Intan RHS data load module by @skim0119 in https://github.com/GazzolaLab/MiV-OS/pull/179
* Support fragment load for large file size.
* MiV native `miv.Spikestamps` datatime with extended 'merge' feature.
* Support for OpenEphys version 0.5.4 file structure.
* Add array type return for spiketrain during spikedetection.
* Add `spike_amplitude_to_background_noise` statistics.
* New protocol for `Data` module prot

## Minor

- bug: zero spikerate
- docs: update on SNR tools
- test: unittests on SNR features
- test: unittests on BSA coding module
- test: unittests on arduino module
- test: unittests on StimJim module
- test: unittests on geometry protocol
- test: unittests on OE TTL event readout
- test: more unittests on butter-bandpass module
- git-action: use python-action cache for poetry
- git-action: remove windows test with cache

## Dependencies

- deprecated: removed h5py_cache
- pyseries -> pyserial dependency

**Full Changelog**: https://github.com/GazzolaLab/MiV-OS/compare/v0.2.2...v0.2.3

# Release Note (version 0.2.2)

## What's Changed

* Import binary raw file by @skim0119 in https://github.com/GazzolaLab/MiV-OS/pull/116
* Include reverse capability for stimjim
* Documentation Update

## Bug Fix

* Binned Spiketrain algorithm now includes counting
* Fix bug in synchronizing timestamps for TTL event recording
* Update dependencies and fix related issues

**Full Changelog**: https://github.com/GazzolaLab/MiV-OS/compare/v0.2.1...v0.2.2

# Release Note (version 0.2.1)

## What's Changed

* Test Cases + Bug fix binned_spiketrain by @Gauravu2 in https://github.com/GazzolaLab/MiV-OS/pull/93
* Shift documentation theme to PyData by @skim0119 in https://github.com/GazzolaLab/MiV-OS/pull/94
* Initial implementation of MiV HDF5-based data format by @iraikov in https://github.com/GazzolaLab/MiV-OS/pull/90
* TTL event readout from OpenEphys dataset by @skim0119 in https://github.com/GazzolaLab/MiV-OS/pull/99
* Add core datatypes: SpikeTrain by @skim0119 in https://github.com/GazzolaLab/MiV-OS/pull/102
* Setup dependabot by @skim0119 in https://github.com/GazzolaLab/MiV-OS/pull/103
* Documentation theme updated to `pydata-theme`

## New Contributors

* @iraikov made their first contribution in https://github.com/GazzolaLab/MiV-OS/pull/90
* @dependabot made their first contribution in https://github.com/GazzolaLab/MiV-OS/pull/107

## Dependency Version Upgrade

* build(deps-dev): bump flake8 from 4.0.1 to 5.0.4 by @dependabot in https://github.com/GazzolaLab/MiV-OS/pull/107
* build(deps): bump codecov/codecov-action from 2 to 3 by @dependabot in https://github.com/GazzolaLab/MiV-OS/pull/105
* build(deps): bump actions/setup-python from 2.2.2 to 4.2.0 by @dependabot in https://github.com/GazzolaLab/MiV-OS/pull/104
* build(deps-dev): bump pylint from 2.14.5 to 2.15.0 by @dependabot in https://github.com/GazzolaLab/MiV-OS/pull/111

**Full Changelog**: https://github.com/GazzolaLab/MiV-OS/compare/v0.2.0...v0.2.1

# Release Note (version 0.2.0)

[Milestone](https://github.com/GazzolaLab/MiV-OS/issues/30)

## What's Changed

* Connectivity visualization by @Gauravu2 in https://github.com/GazzolaLab/MiV-OS/pull/82
* Audio encoding (Lyon's Ear Model + BSA) by @skim0119 in https://github.com/GazzolaLab/MiV-OS/pull/91
* Update Tests

**Full Changelog**: https://github.com/GazzolaLab/MiV-OS/compare/v0.1.2...v0.2.0

# Release Note (version 0.1.2)

## What's Changed

* Feature: automatic channel masking by @jihugo in https://github.com/GazzolaLab/MiV-OS/pull/56
* Connectivity: Spike triggered average plot by @Gauravu2 in https://github.com/GazzolaLab/MiV-OS/pull/71
* Visualization: Multi channel signal replay over MEA geometry (#24, #59) by @armantekinalp in https://github.com/GazzolaLab/MiV-OS/pull/66
* Statistics: Fano factor by @Gauravu2 in https://github.com/GazzolaLab/MiV-OS/pull/67

## Minor Updates

* Refactor: remove poetry redundancies (#74) by @bhosale2 in https://github.com/GazzolaLab/MiV-OS/pull/75
* Update tests by @skim0119 in https://github.com/GazzolaLab/MiV-OS/pull/78

## New Contributors

* @jihugo made their first contribution in https://github.com/GazzolaLab/MiV-OS/pull/56
* @bhosale2 made their first contribution in https://github.com/GazzolaLab/MiV-OS/pull/75

**Full Changelog**: https://github.com/GazzolaLab/MiV-OS/compare/v0.1.1...v0.1.2

# Release Note (version 0.1.1)

## Developer Note

Minor release with statistics and connectivity modules.

## What's Changed
* Patch CONTRIBUTING.md guideline by @skim0119 in https://github.com/GazzolaLab/MiV-OS/pull/55
* Statistics: PSTH implementation by @armantekinalp in https://github.com/GazzolaLab/MiV-OS/pull/57
* Connectivity guide update by @Gauravu2 in https://github.com/GazzolaLab/MiV-OS/pull/53
* Burst analysis for a single channel and plotting across channels by @Gauravu2 in https://github.com/GazzolaLab/MiV-OS/pull/60
* Information Theory Module by @Gauravu2 in https://github.com/GazzolaLab/MiV-OS/pull/63

**Full Changelog**: https://github.com/GazzolaLab/MiV-OS/compare/v0.1.0...v0.1.1

# Release Note (version 0.1.0)

## Developer Note

## What's Changed

* Update: Sample dataset and web-installation utilities by @skim0119 in https://github.com/GazzolaLab/MiV-OS/pull/39
* Statistics: Inter-spike interval histogram and statistics documentation by @armantekinalp in https://github.com/GazzolaLab/MiV-OS/pull/44
* Connectivity - documentation on cross correlation and pattern detection using cell assembly by @Gauravu2 in https://github.com/GazzolaLab/MiV-OS/pull/42

## New Contributors
* @Gauravu2 made their first contribution in https://github.com/GazzolaLab/MiV-OS/pull/42

**Full Changelog**: https://github.com/GazzolaLab/MiV-OS/compare/v0.0.6...v0.1.0

# Release Note (version 0.0.6)

## Developer Note

Includes infrastructure for package and initial set of algorithms.

## Notable Changes

* Github hooks by @armantekinalp in https://github.com/GazzolaLab/MiV-OS/pull/9
* filter module, unittest, and documentation by @skim0119 in https://github.com/GazzolaLab/MiV-OS/pull/14
* spike detection module by @skim0119 in https://github.com/GazzolaLab/MiV-OS/pull/18
* statistics doc page and summarizing modules by @skim0119 in https://github.com/GazzolaLab/MiV-OS/pull/19
* data set interface by @armantekinalp in https://github.com/GazzolaLab/MiV-OS/pull/17
* data load documentation added by @armantekinalp in https://github.com/GazzolaLab/MiV-OS/pull/22
* visualization module, unittests, docs by @skim0119 in https://github.com/GazzolaLab/MiV-OS/pull/21
* spike sorting module documentation by @skim0119 in https://github.com/GazzolaLab/MiV-OS/pull/25
* Data module update by @skim0119 in https://github.com/GazzolaLab/MiV-OS/pull/26
* Io module and its unittest by @skim0119 in https://github.com/GazzolaLab/MiV-OS/pull/27
* DataManger modules by @skim0119 in https://github.com/GazzolaLab/MiV-OS/pull/28
* Adopt pyproject.toml and poetry packaging system by @frthjf in https://github.com/GazzolaLab/MiV-OS/pull/32

## Minor fix

* fix:main yml codecoverage by @armantekinalp in https://github.com/GazzolaLab/MiV-OS/pull/12

## New Contributors
* @frthjf made their first contribution in https://github.com/GazzolaLab/MiV-OS/pull/32

**Full Changelog**: https://github.com/GazzolaLab/MiV-OS/compare/v0.0.1...v0.0.6
