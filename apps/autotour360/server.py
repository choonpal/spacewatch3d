#!/usr/bin/env python3
"""Local 360 viewer, byte-range streaming, and explicit trajectory analysis jobs."""
import argparse
import json
import mimetypes
import re
import subprocess
import threading
import uuid
import webbrowser
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, quote, unquote, urlsplit
from trajectory_jobs import TrajectoryJobs, write_json
from storage import StoragePaths

ROOT = Path(__file__).resolve().parent


def parse_range(header, size):
    """Return inclusive byte bounds. Reject malformed, multi, and unsatisfiable ranges."""
    match = re.fullmatch(r"bytes=(\d*)-(\d*)", header.strip())
    if not match or not size or not any(match.groups()):
        raise ValueError("Invalid range")
    first, last = match.groups()
    if not first:
        length = int(last)
        if length <= 0:
            raise ValueError("Invalid suffix")
        return max(0, size - length), size - 1
    start = int(first)
    end = min(int(last), size - 1) if last else size - 1
    if start >= size or start > end:
        raise ValueError("Unsatisfiable range")
    return start, end


def inspect_media(path, index):
    result = {"name": path.name, "size": path.stat().st_size, "url": f"/media/{index}", "duration": 0}
    try:
        probe = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "json", str(path)], capture_output=True, text=True, timeout=10, check=True)
        result["duration"] = float(json.loads(probe.stdout)["format"]["duration"])
    except (FileNotFoundError, subprocess.SubprocessError, KeyError, ValueError):
        pass  # FFmpeg is optional; browser metadata remains authoritative.
    return result


def make_handler(media_paths, metadata, jobs=None, *, export_dir=None):
    tours_dir = Path(export_dir).resolve() if export_dir is not None else StoragePaths.for_app(ROOT).tours
    class Handler(BaseHTTPRequestHandler):
        def send_json(self, data, code=200, head_only=False):
            payload = json.dumps(data, ensure_ascii=False).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(payload)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            if not head_only:
                self.wfile.write(payload)

        def do_POST(self):
            if jobs is None:
                self.send_json({"error": "이 서버에서는 경로 분석을 사용할 수 없습니다."}, 503)
                return
            origin = self.headers.get("Origin")
            if origin and urlsplit(origin).netloc != self.headers.get("Host"):
                self.send_json({"error": "다른 사이트에서 보낸 분석 요청은 허용하지 않습니다."}, 403)
                return
            route = urlsplit(self.path)
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if route.path == "/api/tours":
                    if not 0 < length <= 20 * 1024**2:
                        raise ValueError("투어 파일은 최대 20MB까지 저장할 수 있습니다.")
                    tour = json.loads(self.rfile.read(length))
                    if not isinstance(tour, dict) or tour.get("version") != 1 or not isinstance(tour.get("assets"), list) or not isinstance(tour.get("scenes"), list):
                        raise ValueError("올바른 투어 데이터가 아닙니다.")
                    title = re.sub(r"[^\w -]", "_", str(tour.get("title", "tour")))[:60].strip() or "tour"
                    name = f"{title}-{datetime.now():%Y%m%d-%H%M%S}-{uuid.uuid4().hex[:8]}.tour.json"
                    write_json(tours_dir / name, tour)
                    self.send_json({"fileName": name, "url": "/exports/" + quote(name)}, 201)
                    return
                if route.path in ("/api/videos", "/api/trajectory/import"):
                    if route.path == "/api/videos" and not jobs.available:
                        raise ValueError("자동 분석 도구가 준비되지 않았습니다. setup-trajectory를 실행한 뒤 서버를 다시 시작해 주세요.")
                    name = parse_qs(route.query).get("name", ["video.mp4"])[0]
                    key = jobs.register_upload(self.rfile, length, name)
                    if route.path == "/api/videos":
                        jobs.start(key, force=jobs.status(key)["state"] in ("failed", "cancelled", "interrupted"))
                    media = next(item for item in jobs.media_catalog() if item["trajectoryKey"] == key)
                    self.send_json({"key": key, "status": jobs.status(key), "media": media}, 201)
                    return
                action = re.fullmatch(r"/api/trajectory/([a-f0-9]{64})/(start|cancel)", route.path)
                if not action:
                    self.send_json({"error": "알 수 없는 분석 요청입니다."}, 404)
                    return
                if not 0 <= length <= 1024:
                    raise ValueError("요청 크기가 올바르지 않습니다.")
                data = json.loads(self.rfile.read(length) or b"{}")
                if not isinstance(data, dict):
                    raise ValueError("분석 요청은 JSON 객체여야 합니다.")
                result = jobs.start(action[1], force=data.get("force") is True) if action[2] == "start" else jobs.cancel(action[1])
                self.send_json(result, 202)
            except KeyError as error:
                self.send_json({"error": str(error)}, 404)
            except (ValueError, OSError) as error:
                self.close_connection = True
                self.send_json({"error": str(error)}, 400)

        def do_HEAD(self):
            self.serve(head_only=True)

        def do_GET(self):
            self.serve(head_only=False)

        def serve(self, head_only=False):
            route = unquote(urlsplit(self.path).path)
            if route in ("/api/config", "/api/library"):
                self.send_json({"media": jobs.media_catalog() if jobs else metadata, "trajectory": {"available": bool(jobs and jobs.available), "maxUploadBytes": TrajectoryJobs.MAX_UPLOAD}}, head_only=head_only)
                return
            trajectory_route = re.fullmatch(r"/api/trajectory/([a-f0-9]{64})(/result)?", route)
            if trajectory_route and jobs:
                try:
                    if trajectory_route[2]:
                        result_path = jobs.result_path(trajectory_route[1])
                        if not result_path:
                            raise KeyError("분석 결과가 없습니다.")
                        self.send_json(json.loads(result_path.read_text()), head_only=head_only)
                    else:
                        self.send_json(jobs.status(trajectory_route[1]), head_only=head_only)
                except KeyError as error:
                    self.send_json({"error": str(error)}, 404, head_only=head_only)
                return
            stored_media = re.fullmatch(r"/media/([a-f0-9]{64})", route)
            export_match = re.fullmatch(r"/exports/([\w -]+\.tour\.json)", route)
            media_match = re.fullmatch(r"/media/(\d+)", route)
            if export_match:
                path = tours_dir / export_match[1]
            elif stored_media and jobs:
                source = jobs.sources.get(stored_media[1])
                if not source:
                    self.send_error(404)
                    return
                path = source["path"]
            elif media_match:
                index = int(media_match.group(1))
                if index >= len(media_paths):
                    self.send_error(404)
                    return
                path = media_paths[index]
            else:
                name = "index.html" if route == "/" else route.lstrip("/")
                # Never expose source video directories, local config, repository, or test files.
                allowed = name in {"index.html", "styles.css", "app.js", "core.js", "panorama.js", "trajectory.js", "trajectory-math.js", "autotour.js", "autotour.css"} or re.fullmatch(r"assets/[a-zA-Z0-9_-]+\.(jpg|png|webp|svg)", name)
                if not allowed:
                    self.send_error(404)
                    return
                path = ROOT / name
            if not path.is_file():
                self.send_error(404, "File unavailable")
                return
            try:
                with path.open("rb") as source:
                    size = path.stat().st_size
                    start, end = 0, size - 1
                    range_header = self.headers.get("Range")
                    if range_header:
                        try:
                            start, end = parse_range(range_header, size)
                        except ValueError:
                            self.send_response(416)
                            self.send_header("Content-Range", f"bytes */{size}")
                            self.send_header("Content-Length", "0")
                            self.end_headers()
                            return
                    self.send_response(206 if range_header else 200)
                    self.send_header("Content-Type", mimetypes.guess_type(path.name)[0] or "application/octet-stream")
                    self.send_header("Content-Length", str(max(0, end - start + 1)))
                    self.send_header("Accept-Ranges", "bytes")
                    self.send_header("Cache-Control", "no-cache")
                    if export_match:
                        self.send_header("Content-Disposition", "attachment; filename=\"auto-tour.tour.json\"; filename*=UTF-8''" + quote(path.name))
                    self.send_header("X-Content-Type-Options", "nosniff")
                    if range_header:
                        self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
                    self.end_headers()
                    if not head_only:
                        source.seek(start)
                        remaining = end - start + 1
                        while remaining > 0:
                            chunk = source.read(min(1024 * 1024, remaining))
                            if not chunk:
                                break
                            self.wfile.write(chunk)
                            remaining -= len(chunk)
            except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
                pass  # Browser cancels streams when seeking or changing scenes.
            except OSError:
                self.close_connection = True

    return Handler


def main():
    parser = argparse.ArgumentParser(description="AUTO TOUR 360 local application")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8766)
    parser.add_argument("--media", action="append", help="Path to a 360 video; may be repeated")
    parser.add_argument("--open", action="store_true", help="Open the browser automatically")
    parser.add_argument("--data-dir", help="Uploaded video directory; relative paths use the app directory")
    parser.add_argument("--output-dir", help="Analysis and tour output directory; relative paths use the app directory")
    args = parser.parse_args()
    entries = args.media
    if entries is None and (ROOT / ".local.json").exists():
        try:
            entries = json.loads((ROOT / ".local.json").read_text()).get("media", [])
        except (OSError, ValueError):
            entries = []
    paths = []
    for entry in entries or []:
        path = Path(entry).expanduser().resolve()
        if path.is_file():
            paths.append(path)
        else:
            print(f"Media unavailable: {path.name}. Select it in the viewer instead.", flush=True)
    metadata = [inspect_media(p, i) for i, p in enumerate(paths)]
    jobs = TrajectoryJobs(ROOT, paths, metadata, data_dir=args.data_dir, output_dir=args.output_dir)
    if jobs.available:
        for meta in metadata:
            jobs.start(meta["trajectoryKey"])
    server = ThreadingHTTPServer((args.host, args.port), make_handler(paths, metadata, jobs, export_dir=jobs.storage.tours))
    server.daemon_threads = True
    url = f"http://{'127.0.0.1' if args.host == '0.0.0.0' else args.host}:{server.server_port}"
    print(f"AUTO TOUR 360: {url}\nLocal media: {len(jobs.sources)} file(s). Trajectory analysis: {'ready' if jobs.available else 'setup required'}.\nVideos: {jobs.storage.data}\nOutputs: {jobs.storage.output}\nPress Ctrl+C to stop.", flush=True)
    if args.open:
        threading.Timer(.5, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        jobs.shutdown()
        server.server_close()


if __name__ == "__main__":
    main()
