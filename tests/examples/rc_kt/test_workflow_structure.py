from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from examples.rc_kt import KT, RC, batch_spontaneous, convert_data_to_h5
from examples.rc_kt.common import load_manifest
from miv.core import Signal
from miv.io import ImportSignal


def test_manifest_is_authoritative_and_rejects_missing_pairing_fields(tmp_path) -> None:
    example = Path("examples/rc_kt/manifest.example.json")
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "manifest.json").write_text(example.read_text())
    manifest = load_manifest(data_dir)
    assert len(manifest["recordings"]) == 4

    del manifest["recordings"][0]["network_id"]
    (data_dir / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="network_id"):
        load_manifest(data_dir)


@pytest.mark.parametrize("source_format", ["intan-rhs", "open-ephys"])
def test_conversion_preserves_aligned_ttl_groups(
    tmp_path, monkeypatch, source_format
) -> None:
    class Loader:
        def load(self):
            yield Signal(
                data=np.arange(16, dtype=float).reshape(4, 4),
                timestamps=np.arange(4) / 30_000,
                rate=30_000.0,
            )
            yield Signal(
                data=np.arange(16, 32, dtype=float).reshape(4, 4),
                timestamps=np.arange(4, 8) / 30_000,
                rate=30_000.0,
            )

    monkeypatch.setattr(
        convert_data_to_h5,
        "_source",
        lambda *_: (Loader(), np.asarray([1 / 30_000, 6 / 30_000])),
    )
    target = tmp_path / f"{source_format}.h5"
    result = convert_data_to_h5.convert_one(
        {
            "entry": {
                "id": source_format,
                "source_path": str(tmp_path / "raw"),
                "source_format": source_format,
                "stimulus_channel": 0,
            },
            "output_path": str(target),
        }
    )
    assert result["status"] == "ok"
    ephys = list(ImportSignal(str(target), group="Ephys").load())
    stimulus = list(ImportSignal(str(target), group="Stimulus").load())
    assert [item.data.shape for item in ephys] == [(4, 4), (4, 4)]
    assert sum(int(item.data.sum()) for item in stimulus) == 2
    for signal, ttl in zip(ephys, stimulus, strict=True):
        np.testing.assert_array_equal(signal.timestamps, ttl.timestamps)


def test_slurm_launchers_use_one_existing_allocation() -> None:
    root = Path("examples/rc_kt")
    for name in (
        "batch_spontaneous.slurm",
        "RC.slurm",
        "KT.slurm",
        "convert_data_to_h5.slurm",
    ):
        content = (root / name).read_text()
        assert "#SBATCH --partition=normal" in content
        assert "#SBATCH --nodes=32" in content
        assert "#SBATCH --ntasks-per-node=1" in content
        assert "#SBATCH --cpus-per-task=4" in content
        assert "#SBATCH --time=48:00:00" in content
        assert "sbatch " not in content
        assert "--n-jobs 32" in content


def test_analysis_modules_import_without_cluster_configuration() -> None:
    assert callable(batch_spontaneous.main)
    assert callable(RC.main)
    assert callable(KT.main)
    assert callable(convert_data_to_h5.main)
