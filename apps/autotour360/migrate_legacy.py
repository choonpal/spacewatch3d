#!/usr/bin/env python3
"""Copy standalone AUTO TOUR 360 data without changing the original installation."""
import argparse
import json
import re
import shutil
import uuid
from datetime import datetime
from pathlib import Path

from storage import StoragePaths
from trajectory_jobs import digest_file, write_json


def copy_checked(source, target):
    if source.is_symlink():
        raise ValueError(f"Symbolic links are not migrated: {source}")
    checksum = digest_file(source)
    if target.exists():
        if not target.is_file() or digest_file(target) != checksum:
            raise ValueError(f"Existing destination differs; no overwrite: {target}")
        return checksum
    target.parent.mkdir(parents=True, exist_ok=True)
    temp = target.with_name(target.name + f".{uuid.uuid4().hex}.part")
    try:
        shutil.copy2(source, temp)
        if digest_file(temp) != checksum:
            raise ValueError(f"Copy verification failed: {source}")
        temp.replace(target)
    finally:
        temp.unlink(missing_ok=True)
    return checksum


def migrate(source_root, storage):
    source_root = Path(source_root).resolve()
    cache = source_root / ".data" / "trajectory"
    if not cache.is_dir():
        raise ValueError(f"Standalone data directory not found: {cache}")
    report = {"source": str(source_root), "videos": [], "files": []}

    def copy(source, target):
        checksum = copy_checked(source, target)
        report["files"].append({"source": str(source), "target": str(target), "sha256": checksum})

    for directory in sorted(cache.iterdir()):
        if not directory.is_dir() or not re.fullmatch(r"[a-f0-9]{64}", directory.name):
            continue
        metadata_path = directory / "source.json"
        if not metadata_path.is_file():
            continue
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        name = metadata["file"]
        if not isinstance(name, str) or name != Path(name).name or "\\" in name:
            raise ValueError(f"Invalid source filename: {metadata_path}")
        video = directory / name
        if video.resolve().parent != directory.resolve() or video.is_symlink():
            raise ValueError(f"Source is outside its video directory: {video}")
        if digest_file(video) != directory.name or video.stat().st_size != metadata["size"]:
            raise ValueError(f"Video identity mismatch: {video}")
        status_path = directory / "status.json"
        if status_path.exists() and json.loads(status_path.read_text()).get("state") in {"queued", "running", "cancelling"}:
            raise ValueError(f"Finish or cancel analysis before migration: {directory.name}")
        result_path = directory / "trajectory.json"
        if result_path.exists() and json.loads(result_path.read_text())["source"]["sha256"] != directory.name:
            raise ValueError(f"Trajectory identity mismatch: {result_path}")
        copy(video, storage.data / directory.name / name)
        copy(metadata_path, storage.data / directory.name / "source.json")
        for file in sorted(directory.rglob("*")):
            if file.is_symlink():
                raise ValueError(f"Symbolic links are not migrated: {file}")
            if file.is_file() and file not in (video, metadata_path):
                copy(file, storage.analysis / directory.name / file.relative_to(directory))
        report["videos"].append({"name": metadata["name"], "sha256": directory.name, "bytes": metadata["size"]})
    for file in sorted((source_root / "exports").glob("*.tour.json")):
        copy(file, storage.tours / file.name)
    report_path = storage.output / f"migration-{datetime.now():%Y%m%d-%H%M%S}-{uuid.uuid4().hex[:8]}.json"
    write_json(report_path, report)
    return report, report_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="Original standalone application directory")
    parser.add_argument("--data-dir")
    parser.add_argument("--output-dir")
    args = parser.parse_args()
    storage = StoragePaths.for_app(Path(__file__).resolve().parent, args.data_dir, args.output_dir)
    report, report_path = migrate(args.source, storage)
    print(f"Verified {len(report['videos'])} videos and {len(report['files'])} files. Report: {report_path}")


if __name__ == "__main__":
    main()
