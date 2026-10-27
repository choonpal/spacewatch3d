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
from mesh_change_jobs import MeshChangeJobs
from mesh_segmentation_review import ReviewAssets

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


def make_handler(media_paths, metadata, jobs=None, *, export_dir=None, evidence_dir=None, rescan_dir=None, mesh_comparison_dir=None, independent_dir=None, rgbd_dir=None, residual_dir=None, transfer_dir=None, motion_assessment_path=None, dorm_dir=None, dorm_tour_dir=None, mesh_change_jobs=None):
    tours_dir = Path(export_dir).resolve() if export_dir is not None else StoragePaths.for_app(ROOT).tours
    evidence_root = Path(evidence_dir).resolve() if evidence_dir is not None else ROOT.parents[1] / "outputs/change_evidence_demo"
    rescan_root = Path(rescan_dir).resolve() if rescan_dir is not None else ROOT.parents[1] / "outputs/rescan-review"
    comparison_root = Path(mesh_comparison_dir).resolve() if mesh_comparison_dir is not None else ROOT.parents[1] / "outputs/mesh-comparison-final2-20261015"
    independent_root = Path(independent_dir).resolve() if independent_dir is not None else ROOT.parents[1] / "outputs/m3c2-noise-screen-v2-20261015"
    rgbd_root = Path(rgbd_dir).resolve() if rgbd_dir is not None else ROOT.parents[1] / "outputs/bonn-removal-20261015/published"
    residual_root = Path(residual_dir).resolve() if residual_dir is not None else ROOT.parents[1] / "outputs/all-surfaces-rescan-v2-20261016/published"
    transfer_root = Path(transfer_dir).resolve() if transfer_dir is not None else ROOT.parents[1] / "outputs/frozen-transfer-bonn-placement-20261016/run-v1/published"
    dorm_root = Path(dorm_dir).resolve() if dorm_dir is not None else ROOT.parents[1] / "outputs/dorm2-dorm3-20261019/alignment-v9/published"
    dorm_tour_root = Path(dorm_tour_dir).resolve() if dorm_tour_dir is not None else ROOT.parents[1] / "outputs/dorm-object-photo-tour-20261023/run-v6/published"
    motion_report = Path(motion_assessment_path).resolve() if motion_assessment_path is not None else ROOT.parents[1] / "outputs/local-motion-expanded-20261019/run-v1/motion-assessment.json"
    segmentation_reviews = ReviewAssets(mesh_change_jobs) if mesh_change_jobs else None
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
            mesh_route = urlsplit(self.path).path
            retry = re.fullmatch(r"/api/mesh-change-jobs/([a-f0-9]{24})/retry", mesh_route)
            source_submit = mesh_route == '/api/mesh-change-jobs/from-captures'
            if mesh_route == "/api/mesh-change-jobs" or retry or source_submit:
                origin = self.headers.get("Origin")
                if origin and urlsplit(origin).netloc != self.headers.get("Host"):
                    self.send_json({"error": "다른 사이트의 분석 요청은 허용하지 않습니다."}, 403)
                    return
                try:
                    if mesh_change_jobs is None:
                        self.send_json({"error": "변화탐지 실행 환경이 설정되지 않았습니다."}, 503)
                        return
                    if source_submit:
                        length = int(self.headers.get('Content-Length', '0'))
                        if not 0 < length <= 4096:
                            raise ValueError('전후 촬영 선택을 확인해 주세요.')
                        result = mesh_change_jobs.submit_sources(json.loads(self.rfile.read(length)))
                    elif retry:
                        if int(self.headers.get("Content-Length", "0")) != 0:
                            raise ValueError('재연결 요청에는 본문이 필요하지 않습니다.')
                        result = mesh_change_jobs.retry(retry[1])
                    else:
                        result = mesh_change_jobs.submit(self.rfile, int(self.headers.get("Content-Length", "0")))
                    self.send_json(result, 202)
                except (KeyError, ValueError, OSError) as error:
                    self.close_connection = True
                    self.send_json({"error": str(error)}, 400)
                return
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
            if route == "/api/mesh-change-jobs/config":
                self.send_json(mesh_change_jobs.configuration() if mesh_change_jobs else {"configured": False, "connected": False}, head_only=head_only)
                return
            if route == '/api/mesh-change-jobs/captures':
                try:
                    self.send_json({'captures': mesh_change_jobs.sources() if mesh_change_jobs else []}, head_only=head_only)
                except (OSError, ValueError) as error:
                    self.send_json({'error': str(error)}, 400, head_only=head_only)
                return
            mesh_job = re.fullmatch(r"/api/mesh-change-jobs/([a-f0-9]{24})(?:/files/(.+))?", route)
            mesh_job_path = None
            review = re.fullmatch(r"/api/mesh-change-jobs/([a-f0-9]{24})/segmentation-review/(.+)", route)
            if review:
                try:
                    if segmentation_reviews is None:
                        raise KeyError('검증할 입력이 없습니다.')
                    mesh_job_path = segmentation_reviews.asset(review[1], review[2])
                except (KeyError, ValueError, OSError) as error:
                    self.send_json({'error': str(error)}, 404, head_only=head_only)
                    return
            if mesh_job:
                try:
                    if mesh_change_jobs is None:
                        raise KeyError("변화탐지 작업이 없습니다.")
                    if not mesh_job[2]:
                        self.send_json(mesh_change_jobs.status(mesh_job[1]), head_only=head_only)
                        return
                    mesh_job_path = mesh_change_jobs.asset(mesh_job[1], mesh_job[2])
                except KeyError as error:
                    self.send_json({"error": str(error)}, 404, head_only=head_only)
                    return
            stored_media = re.fullmatch(r"/media/([a-f0-9]{64})", route)
            export_match = re.fullmatch(r"/exports/([\w -]+\.tour\.json)", route)
            media_match = re.fullmatch(r"/media/(\d+)", route)
            evidence_match = re.fullmatch(r"/evidence/(evidence\.json|before\.bin|after\.bin|region-\d{3}(?:\.indices)?\.bin)", route)
            rescan_match = re.fullmatch(r"/rescan/((?:review|changes)\.json|[01]-(?:object-)?(?:mesh|points)\.bin|frame-[01]-\d{6}\.jpg|region-\d{3}\.indices\.bin)", route)
            comparison_match = re.fullmatch(r"/mesh-comparison/(comparison\.json|[01]-comparison\.bin)", route)
            independent_match = re.fullmatch(r"/m3c2-independent/(changes\.json|[01]-region-ids\.bin|frame-[01]-\d{6}\.jpg)", route)
            rgbd_match = re.fullmatch(r"/datasets/bonn-removal/((?:comparison|reference|changes)\.json|[01]-(?:comparison|region-ids)\.bin|frame-[01]-\d{6}\.jpg)", route)
            residual_match = re.fullmatch(r"/residual-comparison/(comparison\.json|[01]-(?:comparison|fixed|empirical)\.bin|frame-[01]-\d{6}\.jpg)", route)
            transfer_match = re.fullmatch(r"/residual-comparison/bonn-placement/(comparison\.json|[01]-(?:comparison|fixed|empirical)\.bin|frame-[01]-\d{6}\.jpg)", route)
            dorm_match = re.fullmatch(r"/residual-comparison/dorm2-dorm3/((?:comparison|alignment-view)\.json|[01]-(?:comparison|empirical|alignment)\.bin|frame-[01]-\d{6}\.jpg)", route)
            dorm_tour_match = re.fullmatch(r"/dorm-tour/(tour\.json|(?:map|changes)-[01]\.bin|object-(?:E\d{3}-[01]|P\d{4}-[01]|[SI][01]-\d{3})\.bin|(?:Dorm[23]-\d{5}|review-P\d{4})\.jpg|ghost-(?:E\d{3}|P\d{4})-[01]-[01]-\d{2}\.png)", route)
            if mesh_job_path is not None:
                path = mesh_job_path
            elif route == "/rescan/motion-assessment.json":
                path = motion_report
            elif dorm_tour_match:
                path = dorm_tour_root / dorm_tour_match[1]
            elif dorm_match:
                path = dorm_root / dorm_match[1]
            elif transfer_match:
                path = transfer_root / transfer_match[1]
            elif residual_match:
                path = residual_root / residual_match[1]
            elif rgbd_match:
                path = rgbd_root / rgbd_match[1]
            elif independent_match:
                path = independent_root / independent_match[1]
            elif comparison_match:
                path = comparison_root / comparison_match[1]
            elif rescan_match:
                path = rescan_root / rescan_match[1]
            elif evidence_match:
                path = evidence_root / evidence_match[1]
            elif export_match:
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
                allowed = name in {"mesh-changes.html", "mesh-changes.js", "mesh-changes.css", "mesh-input.example.json", "index.html", "styles.css", "app.js", "core.js", "panorama.js", "trajectory.js", "trajectory-math.js", "tour-keyframes.js", "keyframe-controller.js", "keyframe-transition.js", "object-photo.js", "motion-assessment.js", "autotour.js", "autotour.css", "changes.html", "change-evidence.js", "change-evidence.css", "change-tour.html", "change-tour.js", "change-tour.css", "rescan.html", "rescan-review.js", "rescan-review.css", "mesh-comparison.html", "mesh-comparison.js", "mesh-comparison.css", "mesh-change-context.js", "residual-comparison.html", "residual-comparison.js", "residual-comparison.css", "residual-context.js", "dorm-alignment-viewer.js", "dorm-change-tour.js", "dorm-change-tour.css", "dorm-tour-map.js", "dorm-tour-model.js"} or re.fullmatch(r"assets/[a-zA-Z0-9_-]+\.(jpg|png|webp|svg)", name)
                allowed = allowed or name in {'mesh-segmentation.html', 'mesh-segmentation.js', 'mesh-segmentation.css'}
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
    parser.add_argument("--evidence-dir", type=Path, help="Generated change evidence directory (default: repository outputs/change_evidence_demo)")
    parser.add_argument("--rescan-dir", type=Path, help="Real Rescan mesh review directory (default: repository outputs/rescan-review)")
    parser.add_argument("--mesh-comparison-dir", type=Path, help="Geometry-only M3C2 / GLS comparison output directory")
    parser.add_argument("--independent-dir", type=Path, help="Label-free M3C2 change regions, masks and new original keyframes")
    parser.add_argument("--rgbd-dir", type=Path, help="Published Bonn RGB-D comparison and original frames")
    parser.add_argument("--residual-dir", type=Path, help="Optional structural-plane residual comparison experiment")
    parser.add_argument("--transfer-dir", type=Path, help="Published frozen-setting transfer to the unused Bonn placement capture")
    parser.add_argument("--motion-assessment", type=Path, help="Object motion assessment against nearby structural-background residuals")
    parser.add_argument("--dorm-dir", type=Path, help="Published geometry-only Dorm2/Dorm3 aligned mesh comparison")
    parser.add_argument("--dorm-tour-dir", type=Path, help="Adapted Dorm mesh-first maps and frozen automatic event results")
    parser.add_argument("--mesh-change-runtime", type=Path, help="Original RGB SimSaC runtime configuration")
    parser.add_argument("--mesh-change-jobs-dir", type=Path, help="Persistent input jobs and result cache")
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
    mesh_jobs = MeshChangeJobs(args.mesh_change_jobs_dir or ROOT.parents[1]/"outputs/mesh-change-jobs", args.mesh_change_runtime)
    server = ThreadingHTTPServer((args.host, args.port), make_handler(paths, metadata, jobs, export_dir=jobs.storage.tours, evidence_dir=args.evidence_dir, rescan_dir=args.rescan_dir, mesh_comparison_dir=args.mesh_comparison_dir, independent_dir=args.independent_dir, rgbd_dir=args.rgbd_dir, residual_dir=args.residual_dir, transfer_dir=args.transfer_dir, motion_assessment_path=args.motion_assessment, dorm_dir=args.dorm_dir, dorm_tour_dir=args.dorm_tour_dir, mesh_change_jobs=mesh_jobs))
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
