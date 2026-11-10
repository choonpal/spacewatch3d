#!/usr/bin/env python3
"""Read-only, standard-library-only local demo. Does not run any models."""
import argparse
import json
import mimetypes
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import re
from urllib.parse import unquote, urlsplit

ROOT = Path(__file__).resolve().parent


def file_range(header, size):
    if not header:
        return 0, size - 1
    match = re.fullmatch(r'bytes=(\d*)-(\d*)', header)
    if not match or not any(match.groups()) or size <= 0:
        raise ValueError('Invalid byte range')
    first, last = match.groups()
    if not first:
        length = int(last)
        if length <= 0:
            raise ValueError('Invalid suffix range')
        return max(0, size - length), size - 1
    start, end = int(first), min(int(last), size - 1) if last else size - 1
    if not 0 <= start <= end < size:
        raise ValueError('Unsatisfiable byte range')
    return start, end


def make_handler(root, archive=None):
    root = Path(root).resolve()
    release = json.loads((root / 'RELEASE.json').read_text(encoding='utf-8'))
    demo = root / 'demo'
    manifest = json.loads((demo / 'manifest.json').read_text(encoding='utf-8'))
    assets = set(manifest['assets'])
    app = root / 'code/apps/autotour360'
    app_assets = set(release['demo']['app_assets'])

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            self.serve(False)

        def do_HEAD(self):
            self.serve(True)

        def serve(self, head):
            route = unquote(urlsplit(self.path).path)
            if route == '/api/pipeline':
                path = demo / 'manifest.json'
            elif route == '/release.json':
                path = root / 'RELEASE.json'
            elif route == '/download.zip' and archive:
                path = Path(archive)
            elif route.startswith('/pipeline-assets/'):
                name = route[len('/pipeline-assets/'):]
                path = (demo / name).resolve()
                if name not in assets or demo.resolve() not in path.parents:
                    self.send_error(404)
                    return
            else:
                name = 'pipeline.html' if route == '/' else route.lstrip('/')
                path = (app / name).resolve()
                if name not in app_assets or app.resolve() not in path.parents:
                    self.send_error(404)
                    return
            if not path.is_file():
                self.send_error(404)
                return
            size = path.stat().st_size
            try:
                start, end = file_range(self.headers.get('Range'), size)
            except ValueError:
                self.send_response(416)
                self.send_header('Content-Range', f'bytes */{size}')
                self.send_header('Content-Length', '0')
                self.end_headers()
                return
            partial = bool(self.headers.get('Range'))
            self.send_response(206 if partial else 200)
            self.send_header('Content-Type', mimetypes.guess_type(path.name)[0] or 'application/octet-stream')
            self.send_header('Content-Length', str(max(0, end - start + 1)))
            self.send_header('Accept-Ranges', 'bytes')
            self.send_header('Cache-Control', 'no-cache')
            self.send_header('X-Content-Type-Options', 'nosniff')
            if partial:
                self.send_header('Content-Range', f'bytes {start}-{end}/{size}')
            if route == '/download.zip':
                self.send_header('Content-Disposition', 'attachment; filename="SpaceWatch_Final_20261110.zip"')
            self.end_headers()
            if not head:
                try:
                    with path.open('rb') as stream:
                        stream.seek(start)
                        remaining = end - start + 1
                        while remaining > 0:
                            block = stream.read(min(256 * 1024, remaining))
                            if not block:
                                break
                            self.wfile.write(block)
                            remaining -= len(block)
                except (BrokenPipeError, ConnectionResetError):
                    pass
    return Handler


def main():
    parser = argparse.ArgumentParser(description='SpaceWatch final Dorm5 demo; Python standard library only')
    parser.add_argument('--port', type=int, default=8766)
    parser.add_argument('--archive', type=Path, help='Optional explicit ZIP download; no other server files are exposed')
    parser.add_argument('--ready', type=Path)
    args = parser.parse_args()
    if args.archive and (args.archive.suffix != '.zip' or (args.archive.exists() and not args.archive.is_file())):
        parser.error('--archive must name a ZIP file; it may be built after server startup')
    server = ThreadingHTTPServer(('127.0.0.1', args.port), make_handler(ROOT, args.archive))
    url = f'http://127.0.0.1:{server.server_port}'
    if args.ready:
        args.ready.write_text(json.dumps(dict(pid=__import__('os').getpid(), url=url, read_only=True)) + '\n', encoding='utf-8')
    print(url + '/pipeline.html?capture=dorm5#tour', flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == '__main__':
    main()
