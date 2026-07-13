import hashlib
import json
from pathlib import Path

from src.experiments.audit import (
    aggregate_file_digest,
    build_artifact_manifest,
    build_file_hash_manifest,
    collect_repository_state,
    sha256_file,
)


def test_sha256_and_dataset_manifest_are_deterministic(tmp_path: Path) -> None:
    fused_dir = tmp_path / "fused"
    fused_dir.mkdir()
    (fused_dir / "event_0001.npz").write_bytes(b"second")
    (fused_dir / "event_0000.npz").write_bytes(b"first")
    checkpoint = tmp_path / "best.pt"
    checkpoint.write_bytes(b"checkpoint")

    manifest = build_file_hash_manifest(fused_dir, checkpoint, tmp_path)
    assert manifest["dataset"]["event_file_count"] == 2
    assert [row["path"] for row in manifest["dataset"]["files"]] == [
        "fused/event_0000.npz",
        "fused/event_0001.npz",
    ]
    assert manifest == build_file_hash_manifest(fused_dir, checkpoint, tmp_path)
    assert sha256_file(checkpoint) == hashlib.sha256(b"checkpoint").hexdigest()


def test_artifact_manifest_hashes_reloadable_json(tmp_path: Path) -> None:
    names = ("environment.json", "metrics.json")
    for name in names:
        (tmp_path / name).write_text(json.dumps({"name": name}), encoding="utf-8")
    manifest = build_artifact_manifest(tmp_path, names)
    assert manifest["artifact_count"] == 2
    assert manifest["aggregate_sha256"] == aggregate_file_digest(manifest["artifacts"])
    json.loads(json.dumps(manifest))


def test_repository_state_handles_non_git_directory(tmp_path: Path) -> None:
    state = collect_repository_state(tmp_path)
    assert state["git_available"] is False
