"""Fetch a small real Rescan pair from the authors' ZIP using HTTP ranges.

Never downloads the 25 GB archive. Each extracted member is size/CRC checked;
the manifest records the source URL and SHA256 of the unchanged source files.
"""
import argparse
import hashlib
import io
import json
import struct
import urllib.request
import zipfile
import zlib
from pathlib import Path

URL = "https://rescan.cs.princeton.edu/assets/rescan_dataset.zip"


def get_range(start, end):
    request = urllib.request.Request(URL, headers={"Range": f"bytes={start}-{end}"})
    with urllib.request.urlopen(request, timeout=90) as response:
        if response.status != 206:
            raise ValueError("Server did not honor Range; refusing full archive download")
        value = response.read()
        if len(value) != end - start + 1:
            raise ValueError("Truncated HTTP range")
        return value


class RemoteZip(io.RawIOBase):
    def __init__(self, size):
        self.size, self.position = size, 0

    def seekable(self):
        return True

    def seek(self, offset, whence=0):
        self.position = offset if whence == 0 else (self.position if whence == 1 else self.size) + offset
        return self.position

    def tell(self):
        return self.position

    def read(self, size=-1):
        size = min(size if size >= 0 else self.size, self.size - self.position)
        if size <= 0:
            return b""
        value = get_range(self.position, self.position + size - 1)
        self.position += size
        return value


def extract(info, destination):
    header = get_range(info.header_offset, info.header_offset + 29)
    if header[:4] != b"PK\x03\x04" or info.compress_type not in (0, 8):
        raise ValueError("Unsupported ZIP member")
    filename_len, extra_len = struct.unpack_from("<HH", header, 26)
    start = info.header_offset + 30 + filename_len + extra_len
    temporary = destination.with_suffix(destination.suffix + ".partial")
    decoder = zlib.decompressobj(-15) if info.compress_type == 8 else None
    crc, size, digest = 0, 0, hashlib.sha256()
    destination.parent.mkdir(parents=True, exist_ok=True)
    with temporary.open("wb") as stream:
        for offset in range(0, info.compress_size, 4 * 1024 * 1024):
            end = min(offset + 4 * 1024 * 1024, info.compress_size)
            raw = get_range(start + offset, start + end - 1)
            data = decoder.decompress(raw) if decoder else raw
            stream.write(data)
            crc, size = zlib.crc32(data, crc), size + len(data)
            digest.update(data)
            if info.compress_size > 20_000_000 and offset % (32 * 1024 * 1024) == 0:
                print(f"{info.filename}: {end / info.compress_size:.0%}", flush=True)
        if decoder:
            data = decoder.flush()
            stream.write(data)
            crc, size = zlib.crc32(data, crc), size + len(data)
            digest.update(data)
    if size != info.file_size or crc != info.CRC or (decoder and not decoder.eof):
        raise ValueError("Extracted member failed size/CRC validation")
    temporary.replace(destination)
    return {"member": info.filename, "bytes": size, "crc32": f"{crc:08x}", "sha256": digest.hexdigest()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--scene", default="scene_i")
    parser.add_argument("--epochs", nargs=2, type=int, default=[1, 2])
    parser.add_argument("--depth", action="store_true")
    parser.add_argument("--no-color", action="store_true")
    args = parser.parse_args()
    with urllib.request.urlopen(urllib.request.Request(URL, method="HEAD"), timeout=60) as response:
        size = int(response.headers["Content-Length"])
    archive = zipfile.ZipFile(RemoteZip(size))
    names = ["scene_list.txt", "nyu40_classes.txt"]
    for epoch in args.epochs:
        base = f"{args.scene}_{epoch}"
        names.append(f"{args.scene}/gt_segmentation/{base}.ply")
        if not args.no_color:
            names.append(f"{args.scene}/color/{base}.h264")
        ambiguity = f"{args.scene}/gt_segmentation/{base}.txt"
        if ambiguity in archive.namelist():
            names.append(ambiguity)
        if args.depth:
            names.append(f"{args.scene}/depth/{base}.depth")
    args.output.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output / "source.json"
    existing = json.loads(manifest_path.read_text()) if manifest_path.exists() else {"files": []}
    files = []
    for name in names:
        target = args.output / Path(name).name
        cached = next((f for f in existing["files"] if f["member"] == name), None)
        if cached and target.is_file() and hashlib.sha256(target.read_bytes()).hexdigest() == cached["sha256"]:
            files.append(cached)
            print(f"Verified cached {name}", flush=True)
        else:
            files.append(extract(archive.getinfo(name), target))
            print(f"Verified {name}: {files[-1]['bytes']:,} bytes", flush=True)
        manifest_path.write_text(json.dumps({"dataset": "Rescan", "url": URL, "archive_bytes": size,
                                            "scene": args.scene, "epochs": args.epochs,
                                            "files": files}, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
