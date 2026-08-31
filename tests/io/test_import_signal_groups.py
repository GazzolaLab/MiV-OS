from __future__ import annotations

import numpy as np

from miv.io import file as miv_file
from miv.io import ImportSignal


def test_import_signal_group_and_subset(tmp_path) -> None:
    path = tmp_path / "grouped.h5"
    data = miv_file.initialize()
    for group in ("Ephys", "Stimulus"):
        miv_file.create_group(data, group, counter=f"{group}_counter")
        miv_file.create_dataset(data, "Data", group=group, dtype=np.float64)
        miv_file.create_dataset(data, "Timestamps", group=group, dtype=np.float64)
        miv_file.create_dataset(data, "Rate", group=group, dtype=np.float64)
    for container_index in range(3):
        container = miv_file.create_container(data)
        for group in ("Ephys", "Stimulus"):
            container[f"{group}/Data"] = np.full((4, 1), container_index)
            container[f"{group}/Timestamps"] = np.arange(4, dtype=float)
            container[f"{group}/Rate"] = np.array(30_000.0)
        assert miv_file.pack(data, container) == 0
    miv_file.write(str(path), data)

    signals = list(ImportSignal(str(path), group="Stimulus", subset=(1, 3)).load())

    assert len(signals) == 2
    np.testing.assert_array_equal(signals[0].data, np.ones((4, 1)))
    np.testing.assert_array_equal(signals[1].data, np.full((4, 1), 2))
