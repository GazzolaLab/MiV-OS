#!/usr/bin/env python3
"""Convert Intan RHS or Open Ephys recordings to aligned MiV HDF5."""

from __future__ import annotations

import argparse
import json
import time
from itertools import chain
from pathlib import Path
from typing import Any

import h5py
import numpy as np

try:
    from .common import load_manifest, run_tasks, write_summary
except ImportError:  # direct script execution
    from common import load_manifest, run_tasks, write_summary


def _source(source_format: str, source_path: Path, stimulus_channel: int):
    if source_format == "intan-rhs":
        from miv.io.intan import DataIntan

        loader = DataIntan(str(source_path))
        digital = loader.load_digital_in_event()
        if stimulus_channel >= digital.number_of_channels:
            raise IndexError("Intan stimulus_channel exceeds digital input channels")
        event_times = np.asarray(digital[stimulus_channel], dtype=np.float64)
    elif source_format == "open-ephys":
        from miv.io.openephys import Data

        loader = Data(str(source_path))
        ttl = loader.load_ttl_event()
        states = np.asarray(ttl.data).reshape(-1)
        event_times = np.asarray(ttl.timestamps)[states == stimulus_channel + 1]
    else:
        raise ValueError("source_format must be intan-rhs or open-ephys")
    return loader, np.asarray(event_times, dtype=np.float64)


def _initialize_h5(path: Path, channels: int) -> h5py.File:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = h5py.File(path, "w", libver="latest")
    mapping = []
    for group in ("Ephys", "Stimulus"):
        counter = f"{group}/nobj"
        mapping.extend(
            [
                (group, counter),
                (f"{group}/Data", counter),
                (f"{group}/Timestamps", counter),
                (f"{group}/Rate", counter),
            ]
        )
        node = handle.create_group(group)
        node.attrs["counter"] = np.bytes_(counter)
        handle.create_dataset(counter, shape=(0,), maxshape=(None,), dtype=np.int64)
        width = channels if group == "Ephys" else 1
        data = handle.create_dataset(
            f"{group}/Data",
            shape=(0, width),
            maxshape=(None, width),
            chunks=True,
            dtype=np.float32,
        )
        stamps = handle.create_dataset(
            f"{group}/Timestamps",
            shape=(0,),
            maxshape=(None,),
            chunks=True,
            dtype=np.float64,
        )
        rate = handle.create_dataset(
            f"{group}/Rate", shape=(0,), maxshape=(None,), dtype=np.float64
        )
        for dataset in (data, stamps, rate, handle[counter]):
            dataset.attrs["_GROUP_"] = np.bytes_(group)
    handle.create_dataset(
        "_MAP_DATASETS_TO_COUNTERS_", data=np.asarray(mapping, dtype="S256")
    )
    handle.attrs["_NUMBER_OF_CONTAINERS_"] = 0
    return handle


def _append(dataset: h5py.Dataset, values: np.ndarray) -> None:
    old = dataset.shape[0]
    dataset.resize(old + values.shape[0], axis=0)
    dataset[old:] = values


def convert_one(task: dict[str, Any]) -> dict[str, Any]:
    entry = task["entry"]
    started = time.time()
    source_path = Path(entry["source_path"])
    output_path = Path(task["output_path"])
    stimulus_channel = int(entry["stimulus_channel"])
    try:
        loader, event_times = _source(
            entry["source_format"], source_path, stimulus_channel
        )
        iterator = iter(loader.load())
        first = next(iterator)
        channels = first.number_of_channels
        handle = _initialize_h5(output_path, channels)
        previous_timestamp = -np.inf
        containers = 0
        try:
            for signal in chain([first], iterator):
                values = np.asarray(signal.data)
                timestamps = np.asarray(signal.timestamps, dtype=np.float64)
                rate = float(signal.rate)
                if values.ndim != 2 or values.shape[1] != channels:
                    raise ValueError("electrophysiology channel count changed")
                if not np.isclose(rate, 30_000.0):
                    raise ValueError(f"expected 30 kHz electrophysiology, got {rate}")
                if timestamps.size != values.shape[0] or np.any(np.diff(timestamps) <= 0):
                    raise ValueError("timestamps must be aligned and strictly increasing")
                if timestamps[0] <= previous_timestamp:
                    raise ValueError("timestamps are not monotonic across fragments")
                previous_timestamp = float(timestamps[-1])
                stimulus = np.zeros((timestamps.size, 1), dtype=np.float32)
                selected = event_times[
                    (event_times >= timestamps[0]) & (event_times <= timestamps[-1])
                ]
                indices = np.searchsorted(timestamps, selected)
                indices = indices[indices < timestamps.size]
                stimulus[indices, 0] = 1.0
                for group, data in (("Ephys", values), ("Stimulus", stimulus)):
                    _append(handle[f"{group}/Data"], np.asarray(data, dtype=np.float32))
                    _append(handle[f"{group}/Timestamps"], timestamps)
                    _append(handle[f"{group}/Rate"], np.asarray([rate]))
                    _append(handle[f"{group}/nobj"], np.asarray([timestamps.size]))
                containers += 1
            handle.attrs["_NUMBER_OF_CONTAINERS_"] = containers
        finally:
            handle.close()

        from miv.io import ImportSignal

        ephys = list(ImportSignal(str(output_path), group="Ephys").load())
        stimulus = list(ImportSignal(str(output_path), group="Stimulus").load())
        if len(ephys) != containers or len(stimulus) != containers:
            raise ValueError("converted HDF5 failed MiV readability validation")
        if any(np.min(item.data) < 0 or np.max(item.data) > 1 for item in stimulus):
            raise ValueError("TTL values must remain in [0, 1]")
        return {
            "id": entry["id"],
            "status": "ok",
            "runtime_seconds": time.time() - started,
            "headline_metrics": {"containers": containers, "channels": channels},
        }
    except Exception as error:
        return {"id": entry["id"], "status": "failed", "error": repr(error)}


def _tasks(args: argparse.Namespace) -> list[dict[str, Any]]:
    if args.conversion_manifest:
        payload = json.loads(Path(args.conversion_manifest).read_text())
        entries = payload.get("recordings", payload)
    elif args.source_path and args.source_format and args.output_path:
        entries = [
            {
                "id": Path(args.source_path).name,
                "source_path": args.source_path,
                "source_format": args.source_format,
                "stimulus_channel": args.stimulus_channel,
                "path": args.output_path,
            }
        ]
    else:
        entries = [
            entry
            for entry in load_manifest(args.data_dir)["recordings"]
            if entry["included"] and entry.get("source_path")
        ]
    tasks = []
    for entry in entries:
        output = (
            Path(entry["path"])
            if Path(entry["path"]).is_absolute()
            else Path(args.data_dir) / entry["path"]
        )
        tasks.append({"entry": entry, "output_path": str(output)})
    return tasks


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--output-dir", default="results")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-jobs", type=int, default=1)
    parser.add_argument("--conversion-manifest")
    parser.add_argument("--source-path")
    parser.add_argument("--source-format", choices=("intan-rhs", "open-ephys"))
    parser.add_argument("--output-path")
    parser.add_argument("--stimulus-channel", type=int, default=0)
    args = parser.parse_args()
    started = time.time()
    results = run_tasks(
        convert_one, _tasks(args), args.n_jobs, Path(args.output_dir) / "parsl"
    )
    write_summary(
        args.output_dir,
        "convert_data_to_h5",
        vars(args),
        started,
        results,
    )
    if any(item["status"] == "failed" for item in results):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
