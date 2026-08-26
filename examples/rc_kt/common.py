"""Shared manifest, execution, and reporting utilities for RC-KT analyses."""

from __future__ import annotations

import importlib.metadata
import json
import os
import socket
import subprocess
import time
from collections.abc import Callable, Iterable, Sequence
from pathlib import Path
from typing import Any

REQUIRED_FIELDS = {
    "id",
    "path",
    "source_format",
    "network_id",
    "cohort",
    "category",
    "elapsed_hours",
    "role",
    "stimulus_channel",
    "channel_map",
    "expert_id",
    "student_id",
    "included",
}


def load_manifest(data_dir: str | Path) -> dict[str, Any]:
    path = Path(data_dir) / "manifest.json"
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing authoritative manifest: {path}. Copy manifest.example.json "
            "and record every scientific pairing explicitly."
        )
    payload = json.loads(path.read_text())
    if payload.get("schema_version") != "1.0":
        raise ValueError("manifest schema_version must be '1.0'")
    recordings = payload.get("recordings")
    if not isinstance(recordings, list):
        raise ValueError("manifest.recordings must be a list")
    seen: set[str] = set()
    for index, entry in enumerate(recordings):
        missing = REQUIRED_FIELDS - set(entry)
        if missing:
            raise ValueError(f"recording {index} is missing fields: {sorted(missing)}")
        if entry["id"] in seen:
            raise ValueError(f"duplicate recording id: {entry['id']}")
        seen.add(entry["id"])
        if entry["source_format"] not in {"miv-h5", "intan-rhs", "open-ephys"}:
            raise ValueError(f"unsupported source_format for {entry['id']}")
    return payload


def eligible_recordings(
    manifest: dict[str, Any], roles: Iterable[str]
) -> list[dict[str, Any]]:
    allowed = set(roles)
    return [
        entry
        for entry in manifest["recordings"]
        if entry["included"] and entry["role"] in allowed
    ]


def resolve_recording_path(data_dir: str | Path, entry: dict[str, Any]) -> Path:
    path = Path(entry["path"])
    return path if path.is_absolute() else Path(data_dir) / path


def package_versions() -> dict[str, str]:
    versions: dict[str, str] = {}
    for distribution in ("MiV-OS", "miv-state-space", "parsl", "numpy"):
        try:
            versions[distribution] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            versions[distribution] = "not-installed"
    return versions


def write_json(path: str | Path, payload: Any) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n")


def write_summary(
    output_dir: str | Path,
    script: str,
    configuration: dict[str, Any],
    started_at: float,
    results: Sequence[dict[str, Any]],
) -> None:
    failures = [item for item in results if item.get("status") == "failed"]
    completed = [item for item in results if item.get("status") == "ok"]
    write_json(
        Path(output_dir) / script / "summary.json",
        {
            "script": script,
            "configuration": configuration,
            "versions": package_versions(),
            "runtime_seconds": time.time() - started_at,
            "included_recordings": [item.get("id") for item in results],
            "completed": len(completed),
            "failures": failures,
            "headline_metrics": [item.get("headline_metrics", {}) for item in completed],
        },
    )


def _slurm_nodes() -> list[str]:
    node_list = os.environ.get("SLURM_JOB_NODELIST")
    if not node_list:
        return []
    result = subprocess.run(
        ["scontrol", "show", "hostnames", node_list],
        check=True,
        capture_output=True,
        text=True,
    )
    return [line for line in result.stdout.splitlines() if line]


def parsl_config(n_jobs: int, run_dir: str | Path):
    """Attach Parsl to the current allocation; never invoke ``sbatch``."""

    from parsl.config import Config
    from parsl.executors import HighThroughputExecutor, ThreadPoolExecutor

    run_dir = Path(run_dir).resolve()
    nodes = _slurm_nodes()
    if not nodes:
        return Config(
            executors=[
                ThreadPoolExecutor(label="rc-kt", max_threads=max(1, n_jobs))
            ],
            run_dir=str(run_dir),
            retries=0,
        )

    from parsl.channels import LocalChannel, SSHChannel
    from parsl.providers import AdHocProvider

    local_names = {socket.gethostname(), socket.getfqdn()}
    channels = []
    for node in nodes:
        if node in local_names or any(name.startswith(node) for name in local_names):
            channels.append(LocalChannel(script_dir=str(run_dir / "parsl-scripts")))
        else:
            channels.append(
                SSHChannel(hostname=node, script_dir=str(run_dir / "parsl-scripts"))
            )
    return Config(
        executors=[
            HighThroughputExecutor(
                label="rc-kt",
                cores_per_worker=4.0,
                max_workers_per_node=1,
                provider=AdHocProvider(
                    channels=channels,
                    worker_init=os.environ.get("RC_KT_WORKER_INIT", ""),
                ),
            )
        ],
        run_dir=str(run_dir),
        retries=0,
    )


def run_tasks(
    function: Callable[[dict[str, Any]], dict[str, Any]],
    tasks: Sequence[dict[str, Any]],
    n_jobs: int,
    run_dir: str | Path,
) -> list[dict[str, Any]]:
    if n_jobs < 1:
        raise ValueError("n_jobs must be positive")
    if n_jobs == 1:
        return [function(task) for task in tasks]

    import parsl
    from parsl.app.app import python_app

    parsl.load(parsl_config(n_jobs, run_dir))
    app = python_app(function, executors=["rc-kt"])
    futures = [(task, app(task)) for task in tasks]
    results: list[dict[str, Any]] = []
    for task, future in futures:
        try:
            results.append(future.result())
        except Exception as error:  # independent recordings must continue
            results.append(
                {"id": task.get("entry", task).get("id"), "status": "failed", "error": repr(error)}
            )
    parsl.dfk().cleanup()
    parsl.clear()
    return results
