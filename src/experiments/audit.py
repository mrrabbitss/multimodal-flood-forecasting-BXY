from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from typing import Any, Sequence


AUDIT_SCHEMA_VERSION = "baseline_audit_v1"


def utc_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def display_path(path: str | Path, repository_root: str | Path) -> str:
    resolved = Path(path).resolve()
    try:
        return resolved.relative_to(Path(repository_root).resolve()).as_posix()
    except ValueError:
        return str(resolved)


def file_record(path: str | Path, repository_root: str | Path) -> dict[str, Any]:
    resolved = Path(path).resolve()
    return {
        "path": display_path(resolved, repository_root),
        "size_bytes": int(resolved.stat().st_size),
        "sha256": sha256_file(resolved),
    }


def aggregate_file_digest(records: Sequence[dict[str, Any]]) -> str:
    canonical = [
        {"path": str(record["path"]), "size_bytes": int(record["size_bytes"]), "sha256": str(record["sha256"])}
        for record in records
    ]
    payload = json.dumps(canonical, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def build_file_hash_manifest(
    fused_dir: str | Path,
    checkpoint: str | Path,
    repository_root: str | Path,
) -> dict[str, Any]:
    fused_root = Path(fused_dir).resolve()
    event_files = sorted(fused_root.glob("event_*.npz"))
    if not event_files:
        raise FileNotFoundError(f"No event_*.npz files found in {fused_root}")
    records = [file_record(path, repository_root) for path in event_files]
    return {
        "schema_version": AUDIT_SCHEMA_VERSION,
        "checkpoint": file_record(checkpoint, repository_root),
        "dataset": {
            "directory": display_path(fused_root, repository_root),
            "event_file_count": len(records),
            "aggregate_sha256": aggregate_file_digest(records),
            "files": records,
        },
    }


def _package_version(name: str) -> str | None:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def _physical_memory_bytes() -> int | None:
    try:
        import psutil

        return int(psutil.virtual_memory().total)
    except (ImportError, AttributeError):
        pass
    if hasattr(os, "sysconf"):
        try:
            return int(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES"))
        except (OSError, ValueError):
            return None
    return None


def collect_environment() -> dict[str, Any]:
    import torch

    cuda_devices = []
    if torch.cuda.is_available():
        for index in range(torch.cuda.device_count()):
            properties = torch.cuda.get_device_properties(index)
            cuda_devices.append(
                {
                    "index": index,
                    "name": properties.name,
                    "total_memory_bytes": int(properties.total_memory),
                    "compute_capability": f"{properties.major}.{properties.minor}",
                }
            )
    return {
        "schema_version": AUDIT_SCHEMA_VERSION,
        "captured_at_utc": utc_timestamp(),
        "platform": platform.platform(),
        "operating_system": platform.system(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "cpu_logical_count": os.cpu_count(),
        "physical_memory_bytes": _physical_memory_bytes(),
        "python": {
            "version": platform.python_version(),
            "executable": sys.executable,
        },
        "packages": {
            name: _package_version(name)
            for name in ("numpy", "pandas", "matplotlib", "scikit-learn", "torch", "tqdm")
        },
        "torch": {
            "version": torch.__version__,
            "cuda_available": bool(torch.cuda.is_available()),
            "cuda_runtime": torch.version.cuda,
            "cudnn_version": torch.backends.cudnn.version(),
            "devices": cuda_devices,
        },
    }


def _run_git(repository_root: Path, *arguments: str) -> tuple[bool, str]:
    try:
        completed = subprocess.run(
            ["git", *arguments],
            cwd=repository_root,
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
        )
    except FileNotFoundError:
        return False, ""
    return completed.returncode == 0, completed.stdout.strip()


def collect_repository_state(repository_root: str | Path) -> dict[str, Any]:
    root = Path(repository_root).resolve()
    inside, inside_value = _run_git(root, "rev-parse", "--is-inside-work-tree")
    if not inside or inside_value != "true":
        return {
            "schema_version": AUDIT_SCHEMA_VERSION,
            "repository_root": str(root),
            "git_available": False,
        }
    _, commit = _run_git(root, "rev-parse", "HEAD")
    _, branch = _run_git(root, "branch", "--show-current")
    remote_ok, remote = _run_git(root, "remote", "get-url", "origin")
    _, status = _run_git(root, "status", "--porcelain=v1")
    status_lines = status.splitlines() if status else []
    return {
        "schema_version": AUDIT_SCHEMA_VERSION,
        "repository_root": str(root),
        "git_available": True,
        "commit": commit,
        "branch": branch,
        "origin": remote if remote_ok else None,
        "working_tree_clean": not status_lines,
        "status_porcelain": status_lines,
        "captured_before_artifact_write": True,
    }


def build_artifact_manifest(output_dir: str | Path, artifact_names: Sequence[str]) -> dict[str, Any]:
    root = Path(output_dir).resolve()
    records = [file_record(root / name, root) for name in artifact_names]
    return {
        "schema_version": AUDIT_SCHEMA_VERSION,
        "captured_at_utc": utc_timestamp(),
        "artifact_count": len(records),
        "aggregate_sha256": aggregate_file_digest(records),
        "artifacts": records,
    }
