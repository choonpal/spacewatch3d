#!/usr/bin/env python3
"""RGB video, image folder, or .tgz -> COLMAP -> OpenMVS (Linux, Python 3.10+).

Uses the installed command-line tools; no third-party Python packages are needed.
Each stage owns a separate directory so an interrupted stage can be restarted.
"""

import argparse
from contextlib import closing
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import shlex
import shutil
import signal
import sqlite3
import struct
import subprocess
import sys
import tarfile
import tempfile
import time
from datetime import datetime, timezone


STAGES = ("frames", "features", "matching", "sfm", "undistort", "convert", "mvs", "validate")
DIRECTORIES = dict(zip(STAGES, ("frames", "features", "matching", "sfm", "undistort", "openmvs_input", "openmvs", "result")))
TOOLS = ("ffmpeg", "colmap", "InterfaceCOLMAP", "DensifyPointCloud")
SCHEMA = 2
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg"}


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def save_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def binary_count(path):
    with path.open("rb") as handle:
        data = handle.read(8)
    require(len(data) == 8, f"잘못된 COLMAP 모델: {path}")
    return struct.unpack("<Q", data)[0]


def image_files(directory):
    return sorted(p for p in directory.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS)


def select_archive_images(path):
    """Inspect members only; never extract archive paths or follow links."""
    with tarfile.open(path, "r:gz") as archive:
        members = archive.getmembers()
    images = []
    for member in members:
        name = PurePosixPath(member.name)
        require(not name.is_absolute() and ".." not in name.parts,
                f"압축 파일에 올바르지 않은 경로가 있습니다: {member.name}")
        if name.suffix.lower() in IMAGE_EXTENSIONS and not member.isdir():
            images.append(member)
    rgb_roots = {PurePosixPath(m.name).parent for m in images if PurePosixPath(m.name).parent.name.lower() == "rgb"}
    roots = rgb_roots or {PurePosixPath(m.name).parent for m in images}
    require(len(roots) == 1, "압축 안의 RGB 이미지 폴더를 하나로 선택할 수 없습니다. 압축을 풀고 원하는 rgb/ 폴더를 직접 지정하세요.")
    root = next(iter(roots))
    selected = [m for m in images if PurePosixPath(m.name).parent == root]
    require(all(m.isfile() for m in selected), "RGB 이미지는 링크가 아닌 일반 파일이어야 합니다.")
    names = [PurePosixPath(m.name).name for m in selected]
    require(len(names) == len(set(names)), "압축 안에 중복된 RGB 이미지 파일명이 있습니다.")
    return sorted((m.name for m in selected), key=lambda name: PurePosixPath(name).name)


def inspect_ply(source, compatible):
    """Validate generated binary PLY vertices, copying only the RGB type aliases."""
    types = {"char": "b", "int8": "b", "uchar": "B", "uint8": "B",
             "short": "h", "int16": "h", "ushort": "H", "uint16": "H",
             "int": "i", "int32": "i", "uint": "I", "uint32": "I",
             "float": "f", "float32": "f", "double": "d", "float64": "d"}
    with source.open("rb") as handle:
        lines, properties, count, element, endian = [], [], None, None, None
        require(handle.readline() == b"ply\n", f"PLY 헤더 오류: {source}")
        lines.append(b"ply\n")
        for _ in range(1000):
            line = handle.readline()
            require(bool(line), "PLY 헤더가 잘렸습니다.")
            lines.append(line)
            words = line.decode("ascii").split()
            if words[:1] == ["format"]:
                endian = {"binary_little_endian": "<", "binary_big_endian": ">"}.get(words[1])
            elif words[:1] == ["element"]:
                element = words[1]
                if element == "vertex":
                    require(count is None, "중복된 PLY vertex 요소")
                    count = int(words[2])
                else:
                    require(int(words[2]) == 0, "점군 전용 PLY가 필요합니다.")
            elif words[:1] == ["property"] and element == "vertex":
                require(len(words) == 3 and words[1] in types, "지원하지 않는 PLY 점 속성")
                properties.append((words[2], types[words[1]]))
                if words[1] == "uint8" and words[2] in ("red", "green", "blue"):
                    lines[-1] = line.replace(b"uint8", b"uchar", 1)
            elif words == ["end_header"]:
                break
        else:
            raise RuntimeError("PLY 헤더가 너무 깁니다.")
        require(endian and count and properties, "비어 있거나 지원하지 않는 PLY")
        names = [name for name, _ in properties]
        require(all(name in names for name in ("x", "y", "z")), "PLY 좌표 속성 누락")
        finite_indices = [i for i, (name, _) in enumerate(properties) if name in ("x", "y", "z", "nx", "ny", "nz")]
        record = struct.Struct(endian + "".join(kind for _, kind in properties))
        offset = handle.tell()
        require(source.stat().st_size - offset == count * record.size, "PLY 데이터 길이가 점 수와 다릅니다.")
        with compatible.open("wb") as output:
            output.writelines(lines)
            remaining = count
            while remaining:
                n = min(remaining, 16384)
                block = handle.read(n * record.size)
                require(len(block) == n * record.size, "PLY 데이터가 잘렸습니다.")
                for values in record.iter_unpack(block):
                    require(all(math.isfinite(values[i]) for i in finite_indices), "PLY에 NaN/무한 좌표 또는 법선이 있습니다.")
                output.write(block)
                remaining -= n
    return {"dense_points": count, "finite_xyz_and_normals": True, "ply_payload_preserved": True}


def positive_float(value):
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise argparse.ArgumentTypeError("양의 유한한 수를 입력하세요.")
    return number


def positive_int(value):
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("양의 정수를 입력하세요.")
    return number


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("input", type=Path, help="RGB 영상, 이미지 폴더, 또는 rgb/를 포함한 .tgz/.tar.gz")
    parser.add_argument("--output", type=Path, required=True, help="새 결과 폴더 (기존 수동 결과 폴더 사용 금지)")
    parser.add_argument("--fps", type=positive_float, help="영상 전용: 초당 추출 프레임 수 (기본 5)")
    parser.add_argument("--image-step", type=positive_int, default=1, help="이미지/압축 전용: 파일명 정렬 후 N장마다 1장 선택")
    parser.add_argument("--threads", type=positive_int, default=8, help="CPU 스레드 수")
    parser.add_argument("--max-image-size", type=positive_int, default=640, help="보정 이미지 및 MVS 최대 변 길이")
    parser.add_argument("--duration", type=positive_float, help="영상 전용: 시작부터 처리할 초 (생략하면 전체)")
    parser.add_argument("--matcher", choices=("exhaustive", "sequential"), default="exhaustive", help="전체 쌍 또는 순서 기반 매칭")
    parser.add_argument("--min-registered-ratio", type=positive_float, default=0.8, help="선택 모델의 최소 이미지 등록 비율")
    parser.add_argument("--stop-after", choices=STAGES, default="validate", help="여기까지 실행하고 종료")
    parser.add_argument("--resume", action="store_true", help="동일한 입력·설정으로 완료 단계 재사용")
    parser.add_argument("--dry-run", action="store_true", help="명령만 출력; 출력 폴더 생성 및 복원 없음")
    args = parser.parse_args(argv)
    if args.min_registered_ratio > 1:
        parser.error("--min-registered-ratio는 0 초과 1 이하여야 합니다.")
    args.input, args.output = args.input.expanduser().resolve(), args.output.expanduser().resolve()
    if not args.input.is_file() and not args.input.is_dir():
        parser.error(f"입력 파일 또는 폴더가 없습니다: {args.input}")
    args.input_kind = "images" if args.input.is_dir() else ("archive" if args.input.name.lower().endswith((".tgz", ".tar.gz")) else "video")
    if args.output == args.input or args.output in args.input.parents or (args.input_kind == "images" and args.input in args.output.parents):
        parser.error("입력과 출력 폴더는 서로 포함되지 않는 별도의 경로여야 합니다.")
    if args.input_kind != "video" and (args.fps is not None or args.duration is not None):
        parser.error("이미지/압축 입력에는 --fps와 --duration을 사용할 수 없습니다. --image-step을 사용하세요.")
    if args.input_kind == "video":
        if args.image_step != 1:
            parser.error("영상 입력에는 --image-step 대신 --fps를 사용하세요.")
        args.fps = 5.0 if args.fps is None else args.fps
    return args


class Pipeline:
    def __init__(self, args):
        self.args, self.root = args, args.output
        self.d = {stage: self.root / folder for stage, folder in DIRECTORIES.items()}
        self.source_images = []
        if args.input_kind == "images":
            rgb = args.input / "rgb"
            require(not rgb.is_symlink(), "rgb 폴더는 링크가 아닌 일반 폴더여야 합니다.")
            source = rgb if rgb.is_dir() else args.input
            candidates = image_files(source)
            require(not any(p.is_symlink() for p in candidates), "RGB 이미지는 링크가 아닌 일반 파일이어야 합니다.")
            self.source_images = [str(p.relative_to(args.input)) for p in candidates][::args.image_step]
        elif args.input_kind == "archive":
            self.source_images = select_archive_images(args.input)[::args.image_step]
        if args.input_kind != "video":
            require(len(self.source_images) >= 3, "선택된 RGB 이미지가 최소 3장 필요합니다. 폴더와 --image-step을 확인하세요.")
        self.tools = {}
        for name in TOOLS:
            if name == "ffmpeg" and args.input_kind != "video":
                continue
            path = shutil.which(name)
            require(path, f"실행 파일을 PATH에서 찾을 수 없습니다: {name}")
            self.tools[name] = path
        self.state_path = self.root / "pipeline.json"
        self.stage, self.command_index = "", 0

    def commands(self, stage):
        a, d = self.args, self.d
        colmap = self.tools["colmap"]
        if stage == "frames":
            if a.input_kind != "video":
                return []
            command = [self.tools["ffmpeg"], "-nostdin", "-n", "-hide_banner", "-loglevel", "warning", "-i", a.input]
            if a.duration is not None:
                command += ["-t", a.duration]
            return [command + ["-map", "0:v:0", "-vf", f"fps={a.fps:g}", "-c:v", "png", "-pix_fmt", "rgb24", d[stage] / "frame_%06d.png"]]
        if stage == "features":
            return [[colmap, "feature_extractor", "--database_path", d[stage] / "database.db", "--image_path", d["frames"], "--ImageReader.camera_model", "SIMPLE_RADIAL", "--ImageReader.single_camera", 1, "--SiftExtraction.use_gpu", 0, "--SiftExtraction.num_threads", a.threads]]
        if stage == "matching":
            return [[colmap, a.matcher + "_matcher", "--database_path", d[stage] / "database.db", "--SiftMatching.use_gpu", 0, "--SiftMatching.num_threads", a.threads, "--SiftMatching.guided_matching", 1]]
        if stage == "sfm":
            return [[colmap, "mapper", "--database_path", d[stage] / "database.db", "--image_path", d["frames"], "--output_path", d[stage] / "sparse", "--Mapper.num_threads", a.threads]]
        if stage == "undistort":
            return [[colmap, "image_undistorter", "--image_path", d["frames"], "--input_path", d["sfm"] / "model", "--output_path", d[stage], "--output_type", "COLMAP", "--max_image_size", a.max_image_size]]
        if stage == "convert":
            return [[self.tools["InterfaceCOLMAP"], "--working-folder", d[stage], "--input-file", d["undistort"], "--output-file", d[stage] / "scene.mvs", "--image-folder", d["undistort"] / "images", "--max-threads", a.threads]]
        if stage == "mvs":
            return [[self.tools["DensifyPointCloud"], "--working-folder", d[stage], "--input-file", d["convert"] / "scene.mvs", "--output-file", d[stage] / "scene_dense.mvs", "--archive-type", 2, "--resolution-level", 0, "--max-resolution", a.max_image_size, "--min-resolution", min(320, a.max_image_size), "--number-views", 5, "--iters", 3, "--geometric-iters", 2, "--number-views-fuse", 2, "--fusion-mode", 0, "--fusion-filter", 2, "--estimate-colors", 2, "--estimate-normals", 2, "--max-threads", a.threads, "--tower-mode", 0, "--estimate-roi", 0, "--crop-to-roi", 0, "--remove-dmaps", 0, "--dense-config-file", d[stage] / "densify.cfg"]]
        return [[self.tools["DensifyPointCloud"], "--working-folder", d[stage], "--input-file", d["mvs"] / "scene_dense.mvs", "--output-view-neighbors-file", d[stage] / "view_neighbors.txt", "--tower-mode", 0, "--estimate-roi", 0, "--max-threads", a.threads]]

    def run_command(self, command):
        command = list(map(str, command))
        self.command_index += 1
        log_path = self.root / "logs" / f"{self.stage}_{self.command_index:02d}.log"
        started = time.monotonic()
        print(f"  $ {shlex.join(command)}\n  로그: {log_path}", flush=True)
        with log_path.open("a", encoding="utf-8") as log:
            log.write(f"\n[{datetime.now(timezone.utc).isoformat()}] {shlex.join(command)}\n")
            log.flush()
            process = subprocess.Popen(command, cwd=self.d[self.stage], stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            try:
                while True:
                    try:
                        code = process.wait(timeout=15)
                        break
                    except subprocess.TimeoutExpired:
                        print(f"  {self.stage}: {time.monotonic() - started:.0f}초 경과", flush=True)
            except BaseException:
                if process.poll() is None:
                    try:
                        os.killpg(process.pid, signal.SIGTERM)
                    except ProcessLookupError:
                        pass
                    try:
                        process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        os.killpg(process.pid, signal.SIGKILL)
                        process.wait()
                raise
            log.write(f"\nExit code: {code}; elapsed: {time.monotonic() - started:.3f}s\n")
        require(code == 0, f"{self.stage} 실행 실패 (종료 코드 {code}). 로그: {log_path}")

    def preflight(self):
        # COLMAP 3.7 flags changed in newer releases: fail before starting work.
        with tempfile.TemporaryDirectory(prefix="video-mvs-check-") as temporary:
            for name, flag in (("feature_extractor", "SiftExtraction.use_gpu"), ("exhaustive_matcher", "SiftMatching.use_gpu")):
                result = subprocess.run([self.tools["colmap"], name, "-h"], cwd=temporary, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, timeout=30)
                require(result.returncode == 0 and flag in result.stdout, "설치된 COLMAP CLI가 이 스크립트와 호환되지 않습니다. 검증 환경은 COLMAP 3.7입니다.")

    def configuration(self):
        a = self.args
        source = {"kind": a.input_kind, "path": str(a.input)}
        if a.input_kind == "images":
            source["images"] = {name: sha256(a.input / name) for name in self.source_images}
        else:
            source["sha256"] = sha256(a.input)
            if a.input_kind == "archive":
                source["images"] = self.source_images
        return {"schema": SCHEMA, "script_sha256": sha256(Path(__file__)),
                "input": source, "output": str(self.root),
                "fps": a.fps, "image_step": a.image_step, "threads": a.threads, "max_image_size": a.max_image_size,
                "duration": a.duration, "matcher": a.matcher, "min_registered_ratio": a.min_registered_ratio,
                "tools": {name: {"path": path, "sha256": sha256(Path(path))} for name, path in self.tools.items()}}

    def prepare_images(self):
        destination = self.d["frames"]
        expected = self.state["configuration"]["input"]
        if self.args.input_kind == "images":
            for name in self.source_images:
                target = destination / Path(name).name
                shutil.copy2(self.args.input / name, target)
                require(sha256(target) == expected["images"][name], f"복사 중 입력 이미지가 변경됐습니다: {name}")
        else:
            remaining = set(self.source_images)
            # Stream members and write only selected basenames, never extractall.
            with tarfile.open(self.args.input, "r|gz") as archive:
                for member in archive:
                    if member.name not in remaining:
                        continue
                    require(member.isfile(), f"일반 이미지 파일이 아닙니다: {member.name}")
                    with closing(archive.extractfile(member)) as source:
                        with (destination / PurePosixPath(member.name).name).open("wb") as target:
                            shutil.copyfileobj(source, target)
                    remaining.remove(member.name)
            require(not remaining and sha256(self.args.input) == expected["sha256"], "이미지 준비 중 압축 입력이 변경됐습니다.")
        manifest = [{"source": name, "image": PurePosixPath(name).name} for name in self.source_images]
        save_json(destination / "input_images.json", manifest)
        print(f"  RGB 이미지 {len(manifest)}장 준비; FFmpeg 프레임 추출 생략", flush=True)

    def verify_artifacts(self, record):
        for relative, digest in record["artifacts"].items():
            path = self.root / relative
            require(path.is_file() and sha256(path) == digest, f"완료 단계의 파일이 없거나 변경됐습니다: {path}. 새 출력 폴더로 실행하세요.")

    def execute_stage(self, stage):
        d = self.d
        if stage == "frames" and self.args.input_kind != "video":
            self.prepare_images()
        if stage == "matching":
            # Keep feature extraction immutable; restart matching from its own copy.
            shutil.copy2(d["features"] / "database.db", d[stage] / "database.db")
        if stage == "sfm":
            # COLMAP mapper opens the DB for writing, even when the matches are
            # already present. Protect the matching checkpoint from this write.
            shutil.copy2(d["matching"] / "database.db", d[stage] / "database.db")
            (d[stage] / "sparse").mkdir()
        for command in self.commands(stage):
            self.run_command(command)

        if stage == "frames":
            frames = image_files(d[stage])
            require(len(frames) >= 3, "최소 3장의 프레임이 필요합니다. 영상 길이와 --fps를 확인하세요.")
            return {"frames": len(frames), "input_kind": self.args.input_kind}
        if stage in ("features", "matching"):
            with closing(sqlite3.connect(f"{(d[stage] / 'database.db').as_uri()}?mode=ro", uri=True)) as database:
                table = "keypoints" if stage == "features" else "two_view_geometries"
                count = database.execute(f"SELECT COUNT(*) FROM {table} WHERE rows > 0").fetchone()[0]
            require(count > 0, f"{stage}: 유효한 특징점/매칭을 얻지 못했습니다.")
            return {"nonempty_records": count}
        if stage == "sfm":
            models = [p for p in (d[stage] / "sparse").iterdir() if p.is_dir() and (p / "images.bin").is_file()]
            require(models, "SfM 모델 생성 실패: 장면 겹침, 흔들림, 카메라 이동을 확인하세요.")
            selected = max(sorted(models), key=lambda p: (binary_count(p / "images.bin"), binary_count(p / "points3D.bin")))
            registered, points = binary_count(selected / "images.bin"), binary_count(selected / "points3D.bin")
            frames = self.state["stages"]["frames"]["metrics"]["frames"]
            require(registered >= 3 and points > 0 and registered / frames >= self.args.min_registered_ratio,
                    f"가장 큰 모델에 {registered}/{frames}장만 등록됐습니다. 촬영/프레임 간격을 확인하세요. 기준을 바꾸려면 새 출력 폴더를 사용하세요.")
            (d[stage] / "model").mkdir()
            (d[stage] / "text").mkdir()
            for name in ("cameras.bin", "images.bin", "points3D.bin"):
                shutil.copy2(selected / name, d[stage] / "model" / name)
            colmap = self.tools["colmap"]
            self.run_command([colmap, "model_analyzer", "--path", d[stage] / "model"])
            for kind, target in (("TXT", d[stage] / "text"), ("PLY", d[stage] / "sparse.ply")):
                self.run_command([colmap, "model_converter", "--input_path", d[stage] / "model", "--output_path", target, "--output_type", kind])
            return {"models": len(models), "selected_model": selected.name, "registered_images": registered, "input_images": frames, "sparse_points": points}
        if stage == "undistort":
            count = len(image_files(d[stage] / "images"))
            require(count == self.state["stages"]["sfm"]["metrics"]["registered_images"], "왜곡 보정 이미지 수가 등록 이미지 수와 다릅니다.")
            return {"images": count}
        if stage == "convert":
            path = d[stage] / "scene.mvs"
            require(path.is_file() and path.stat().st_size > 0, "scene.mvs 생성 실패")
            return {"scene_bytes": path.stat().st_size}
        if stage == "mvs":
            for name in ("scene_dense.mvs", "scene_dense.ply"):
                path = d[stage] / name
                require(path.is_file() and path.stat().st_size > 0, f"MVS 결과 누락: {path}")
            require(list(d[stage].glob("depth*.dmap")), "깊이 지도가 생성되지 않았습니다.")
            return {}
        metrics = inspect_ply(d["mvs"] / "scene_dense.ply", d[stage] / "scene_dense_colmap.ply")
        depth_maps = list(d["mvs"].glob("depth*.dmap"))
        require(all(p.stat().st_size > 0 for p in depth_maps), "빈 깊이 지도 파일이 있습니다.")
        sfm = self.state["stages"]["sfm"]["metrics"]
        require(0 < len(depth_maps) <= sfm["registered_images"], "깊이 지도 수가 올바르지 않습니다.")
        report = {**sfm, **metrics, "depth_maps": len(depth_maps), "input_kind": self.args.input_kind,
                  "warnings": [], "units": "arbitrary SfM scale; not meters",
                  "validation_scope": "PLY length and finite coordinates/normals; depth file count/nonempty; MVS model reload. No ground-truth accuracy or depth-pixel validation.",
                  "stage_seconds": {s: r["elapsed_seconds"] for s, r in self.state["stages"].items() if r["status"] == "complete"},
                  "dense_ply": str(d["mvs"] / "scene_dense.ply"), "dense_mvs": str(d["mvs"] / "scene_dense.mvs")}
        if len(depth_maps) != sfm["registered_images"]:
            report["warnings"].append("일부 등록 이미지의 깊이 지도가 생성되지 않았습니다.")
        if sfm["registered_images"] != sfm["input_images"]:
            report["warnings"].append("가장 큰 SfM 모델에 포함된 이미지만 조밀 복원했습니다.")
        save_json(d[stage] / "summary.json", report)
        return metrics

    def execute(self):
        if self.args.dry_run:
            for stage in STAGES:
                print(f"[{stage}]")
                if stage == "frames" and self.args.input_kind != "video":
                    print(f"  # RGB 이미지 {len(self.source_images)}장을 원래 파일명으로 준비 (FFmpeg 생략)")
                for command in self.commands(stage):
                    print(shlex.join(list(map(str, command))))
                if stage == "sfm":
                    print("  # 가장 큰 모델 선택 → sfm/model 복사 → model_analyzer 및 TXT/PLY 내보내기")
                if stage == self.args.stop_after:
                    break
            return
        self.preflight()
        configuration = self.configuration()
        self.root.mkdir(parents=True, exist_ok=True)
        # The OS releases this lock even after an interrupted or killed process.
        with (self.root / ".pipeline.lock").open("a") as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as error:
                raise RuntimeError("같은 출력 폴더에서 다른 파이프라인이 실행 중입니다.") from error
            if self.args.resume:
                require(self.state_path.is_file(), "이 스크립트의 pipeline.json이 없습니다. 새 폴더로 실행하세요.")
                self.state = json.loads(self.state_path.read_text(encoding="utf-8"))
                require(self.state["configuration"] == configuration, "입력·설정·코드·실행 도구가 이전 실행과 다릅니다. 새 출력 폴더를 지정하세요.")
                for record in self.state["stages"].values():
                    if record["status"] == "complete":
                        self.verify_artifacts(record)
            else:
                require(not any(p.name != ".pipeline.lock" for p in self.root.iterdir()), "출력 폴더가 비어 있지 않습니다. 재개는 --resume, 새 실행은 다른 --output을 사용하세요.")
                self.state = {"configuration": configuration, "stages": {}}
                save_json(self.state_path, self.state)
            (self.root / "logs").mkdir(exist_ok=True)
            for stage in STAGES:
                if self.state["stages"].get(stage, {}).get("status") == "complete":
                    print(f"[{stage}] 완료 결과 재사용", flush=True)
                else:
                    self.stage, self.command_index = stage, 0
                    directory = self.d[stage]
                    require(not directory.is_symlink() and directory.resolve().parent == self.root, "단계 폴더는 출력 폴더 내부의 일반 디렉터리여야 합니다.")
                    if directory.exists():
                        shutil.rmtree(directory)
                    directory.mkdir()
                    record = {"status": "running", "started_at": datetime.now(timezone.utc).isoformat()}
                    self.state["stages"][stage] = record
                    save_json(self.state_path, self.state)
                    started = time.monotonic()
                    print(f"[{stage}] 시작", flush=True)
                    try:
                        metrics = self.execute_stage(stage)
                        artifacts = {str(p.relative_to(self.root)): sha256(p) for p in sorted(directory.rglob("*")) if p.is_file()}
                        record.update(status="complete", metrics=metrics, artifacts=artifacts)
                    except BaseException as error:
                        record.update(status="failed", error=str(error) or type(error).__name__)
                        raise
                    finally:
                        record["elapsed_seconds"] = round(time.monotonic() - started, 3)
                        save_json(self.state_path, self.state)
                    print(f"[{stage}] 완료 ({record['elapsed_seconds']:.1f}초)", flush=True)
                if stage == self.args.stop_after:
                    break
        print(f"처리 완료: {self.root}", flush=True)


def main(argv=None):
    try:
        Pipeline(parse_args(argv)).execute()
        return 0
    except KeyboardInterrupt:
        print("\n중단했습니다. 동일 명령에 --resume을 추가하면 완료 단계 다음부터 재개합니다.", file=sys.stderr)
        return 130
    except (RuntimeError, OSError, ValueError, tarfile.TarError, EOFError, subprocess.TimeoutExpired, sqlite3.Error) as error:
        print(f"오류: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt()))
    sys.exit(main())
