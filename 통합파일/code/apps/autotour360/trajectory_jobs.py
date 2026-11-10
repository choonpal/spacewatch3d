"""Local trajectory jobs. Only explicitly registered or uploaded video files are read."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import threading
import time
import uuid
from storage import StoragePaths


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + f".{uuid.uuid4().hex}.tmp")
    temp.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    temp.replace(path)


def digest_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class TrajectoryJobs:
    MAX_UPLOAD = 2 * 1024**3

    def __init__(self, root, paths=(), metadata=(), *, data_dir=None, output_dir=None):
        self.root = Path(root).resolve()
        self.storage = StoragePaths.for_app(self.root, data_dir, output_dir)
        self.data = self.storage.data
        self.cache = self.storage.analysis
        self.data.mkdir(parents=True, exist_ok=True)
        self.cache.mkdir(parents=True, exist_ok=True)
        candidates = [self.root / ".venv" / "bin" / "python", self.root / ".venv" / "Scripts" / "python.exe"]
        self.python = next((str(p) for p in candidates if p.is_file()), sys.executable)
        try:
            probe = subprocess.run([self.python, "-c", "import pycolmap, cv2, numpy, scipy; assert hasattr(pycolmap.CameraModelId, 'EQUIRECTANGULAR')"], capture_output=True, timeout=15)
            self.available = probe.returncode == 0 and all(shutil.which(name) for name in ("ffmpeg", "ffprobe"))
        except (OSError, subprocess.TimeoutExpired):
            self.available = False
        self.sources = {}
        self.lock = threading.RLock()
        self.worker_lock = threading.Lock()
        self.running = {}
        self.cancelled = set()
        self.bundled = {}
        for path in (self.root / "assets").glob("*.trajectory.json"):
            try:
                data = json.loads(path.read_text())
                self.bundled[data["source"]["sha256"]] = path
            except (OSError, ValueError, KeyError):
                continue
        for path, meta in zip(paths, metadata):
            key = digest_file(path)
            self.sources[key] = {"path": path, "name": path.name, "size": path.stat().st_size, "duration": meta.get("duration", 0)}
            meta["trajectoryKey"] = key
        # Reopen previously analyzed browser-selected files after server restart.
        for path in self.data.glob("*/source.json"):
            try:
                saved = json.loads(path.read_text())
                local = path.parent / saved["file"]
                if local.parent == path.parent and local.is_file():
                    self.sources.setdefault(path.parent.name, {**saved, "path": local, "size": local.stat().st_size})
            except (OSError, ValueError, KeyError):
                continue

    def inspect_video(self, path):
        try:
            result = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=width,height:format=duration", "-of", "json", str(path)], capture_output=True, text=True, check=True, timeout=30)
            data = json.loads(result.stdout)
            stream = data["streams"][0]
            width, height = int(stream["width"]), int(stream["height"])
            duration = float(data["format"]["duration"])
        except (OSError, subprocess.SubprocessError, ValueError, KeyError, IndexError):
            raise ValueError("영상 정보를 읽지 못했습니다. 정상적인 MP4/H.264 영상을 선택해 주세요.") from None
        if not height or abs(width / height - 2) > .03:
            raise ValueError("2:1 비율의 스티칭된 360도 영상이 필요합니다. 일반 영상이나 듀얼 어안 원본은 분석할 수 없습니다.")
        if not 3 <= duration <= 600:
            raise ValueError("3초 이상, 10분 이하의 영상을 선택해 주세요.")
        return {"duration": duration, "width": width, "height": height}

    def media_catalog(self):
        with self.lock:
            sources = list(self.sources.items())
        return [{"trajectoryKey": key, "url": f"/media/{key}", "name": source["name"], "size": source["size"], "duration": source.get("duration", 0), "width": source.get("width", 0), "height": source.get("height", 0), "status": self.status(key)} for key, source in sources]

    def result_path(self, key):
        if key not in self.sources:
            return None
        path = self.cache / key / "trajectory.json"
        if path.is_file():
            return path
        return self.bundled.get(key)

    def status(self, key):
        if key not in self.sources:
            raise KeyError("등록되지 않은 영상입니다.")
        path = self.cache / key / "status.json"
        with self.lock:
            running = key in self.running
        if path.exists():
            try:
                status = json.loads(path.read_text())
                if status.get("state") in ("queued", "running", "cancelling") and not running:
                    status = {"state": "interrupted", "message": "분석이 중단되었습니다. 다시 실행할 수 있습니다.", "progress": 0}
                if running or status.get("state") in ("failed", "cancelled", "interrupted"):
                    return {**status, "key": key, "hasResult": bool(self.result_path(key))}
            except (OSError, ValueError):
                pass
        if self.result_path(key):
            return {"state": "complete", "key": key, "progress": 100, "message": "저장된 촬영 경로를 불러왔습니다.", "resultUrl": f"/api/trajectory/{key}/result"}
        return {"state": "idle", "key": key, "progress": 0, "message": "자동 경로 추정을 시작할 수 있습니다."}

    def register_upload(self, stream, length, name):
        if not 0 < length <= self.MAX_UPLOAD:
            raise ValueError("분석용 영상은 최대 2GB까지 전달할 수 있습니다.")
        suffix = Path(name).suffix.lower()
        if suffix not in {".mp4", ".webm", ".mov", ".m4v", ".ogv"}:
            raise ValueError("MP4, WebM 등 영상 파일을 선택해 주세요.")
        temp = self.data / f"upload-{uuid.uuid4().hex}.part"
        digest = hashlib.sha256()
        try:
            with temp.open("wb") as target:
                remaining = length
                while remaining:
                    chunk = stream.read(min(1024 * 1024, remaining))
                    if not chunk:
                        raise ValueError("영상 전달이 중단되었습니다.")
                    target.write(chunk)
                    digest.update(chunk)
                    remaining -= len(chunk)
            details = self.inspect_video(temp)
            key = digest.hexdigest()
            with self.lock:
                if key not in self.sources:
                    directory = self.data / key
                    directory.mkdir(exist_ok=True)
                    local = directory / ("input" + suffix)
                    temp.replace(local)
                    safe_name = name.replace("\\", "/").split("/")[-1][:255]
                    self.sources[key] = {"path": local, "name": safe_name, "size": length, **details}
                    write_json(directory / "source.json", {"file": local.name, "name": safe_name, "size": length, **details})
                (self.cache / key).mkdir(exist_ok=True)
            return key
        finally:
            temp.unlink(missing_ok=True)

    def start(self, key, force=False):
        if key not in self.sources:
            raise KeyError("등록되지 않은 영상입니다.")
        if not self.available:
            raise ValueError("경로 분석 도구가 없습니다. setup-trajectory.sh 또는 setup-trajectory.bat을 실행해 주세요.")
        with self.lock:
            if key in self.running:
                return self.status(key)
            if self.result_path(key) and not force:
                return self.status(key)
            self.cancelled.discard(key)
            self.running[key] = None
            write_json(self.cache / key / "status.json", {"state": "queued", "progress": 0, "message": "분석 대기 중"})
            threading.Thread(target=self._run, args=(key,), daemon=True).start()
        return self.status(key)

    def cancel(self, key):
        with self.lock:
            if key not in self.running:
                return self.status(key)
            self.cancelled.add(key)
            process = self.running[key]
            if process and process.poll() is None:
                if os.name == "posix":
                    os.killpg(process.pid, signal.SIGTERM)
                else:
                    process.terminate()
            write_json(self.cache / key / "status.json", {"state": "cancelling", "progress": 0, "message": "분석을 취소하는 중입니다."})
        return self.status(key)

    def _run(self, key):
        directory = self.cache / key
        try:
            with self.worker_lock:
                with self.lock:
                    if key in self.cancelled:
                        return
                source = self.sources[key]
                command = [self.python, str(self.root / "estimate_trajectory.py"), str(source["path"]), "--output", str(directory / "trajectory.json"), "--work", str(directory / "work"), "--status", str(directory / "status.json")]
                with (directory / "analysis.log").open("w") as log:
                    process = subprocess.Popen(command, stdout=log, stderr=log, start_new_session=os.name == "posix", env={**os.environ, "OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "6"})
                    with self.lock:
                        self.running[key] = process
                        if key in self.cancelled:
                            self.cancel(key)
                    try:
                        code = process.wait(timeout=1800)
                    except subprocess.TimeoutExpired:
                        if os.name == "posix":
                            os.killpg(process.pid, signal.SIGTERM)
                        else:
                            process.terminate()
                        try:
                            process.wait(timeout=15)
                        except subprocess.TimeoutExpired:
                            if os.name == "posix":
                                os.killpg(process.pid, signal.SIGKILL)
                            else:
                                process.kill()
                            process.wait(timeout=5)
                        raise ValueError("분석 제한 시간(30분)을 초과했습니다. 더 짧은 영상을 사용해 주세요.")
                if key in self.cancelled:
                    write_json(directory / "status.json", {"state": "cancelled", "progress": 0, "message": "분석을 취소했습니다."})
                elif code != 0:
                    # Preserve the human-readable error emitted by the estimator.
                    state = json.loads((directory / "status.json").read_text())
                    if state.get("state") != "failed":
                        raise ValueError("분석 도중 오류가 발생했습니다. analysis.log에서 원인을 확인할 수 있습니다.")
                else:
                    result = json.loads((directory / "trajectory.json").read_text())
                    result["source"]["name"] = source["name"]
                    write_json(directory / "trajectory.json", result)
        except Exception as error:
            write_json(directory / "status.json", {"state": "failed", "progress": 0, "message": str(error)})
        finally:
            with self.lock:
                if key in self.cancelled:
                    write_json(directory / "status.json", {"state": "cancelled", "progress": 0, "message": "분석을 취소했습니다."})
                self.running.pop(key, None)

    def shutdown(self):
        for key in list(self.running):
            self.cancel(key)
