"""Portable, bounded capture bundles; only declared source inputs are accepted."""
import hashlib
import json
from pathlib import Path, PurePosixPath
import zipfile


MAX_BYTES = 8 * 1024**3
MAX_FILES = 20000


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(4 * 1024**2), b''):
            h.update(block)
    return h.hexdigest()


def local_file(root, name):
    if not isinstance(name, str) or '\\' in name or ':' in name:
        raise ValueError('입력 경로는 ZIP 내부의 상대 경로여야 합니다.')
    rel = PurePosixPath(name)
    if rel.is_absolute() or '..' in rel.parts:
        raise ValueError('입력 묶음 밖의 경로를 사용할 수 없습니다.')
    target = (root / name).resolve()
    if not target.is_relative_to(root.resolve()) or not target.is_file():
        raise ValueError(f'입력 파일을 찾을 수 없습니다: {name}')
    return target


def inspect_bundle(root):
    root = Path(root).resolve()
    doc = json.loads((root / 'manifest.json').read_text(encoding='utf-8-sig'))
    if doc.get('schema') != 'mesh-simsac-input-v1':
        raise ValueError('manifest.json은 mesh-simsac-input-v1 형식이어야 합니다.')
    paths = {'manifest.json'}
    for phase in ['before', 'after']:
        spec = doc.get(phase, {})
        if not spec.get('mesh') or not spec.get('frames'):
            raise ValueError(f'{phase}: 분할 메쉬와 카메라 정보가 있는 RGB 프레임이 필요합니다.')
        if 'gaussian' in spec:
            raise ValueError('이 분석은 분할 메쉬와 원본 RGB를 사용합니다.')
        for field in ['mesh', 'pointcloud', 'objects', 'video']:
            if field in spec:
                local_file(root, spec[field]); paths.add(spec[field])
        if Path(spec['mesh']).suffix.lower() != '.npz':
            raise ValueError('분할 메쉬 NPZ에 vertices, faces, face_instance_id를 넣어 주세요.')
        for frame in spec['frames']:
            if 'world_from_camera' not in frame or ('image' not in frame and 'video' not in spec):
                raise ValueError('각 RGB 프레임에 촬영 위치·방향과 이미지 경로가 필요합니다.')
            for field in ['image', 'person_mask', 'segmentation']:
                if field in frame:
                    local_file(root, frame[field]); paths.add(frame[field])
    for field in ['context', 'partition']:
        if doc.get('regional_alignment'):
            name = doc['regional_alignment'][field]
            local_file(root, name); paths.add(name)
    return doc, {name: digest(local_file(root, name)) for name in sorted(paths)}


def unpack(archive, destination):
    destination = Path(destination).resolve()
    destination.mkdir(parents=True, exist_ok=False)
    with zipfile.ZipFile(archive) as z:
        entries = z.infolist()
        if len(entries) > MAX_FILES or sum(e.file_size for e in entries) > MAX_BYTES:
            raise ValueError('입력 묶음의 파일 수 또는 압축 해제 크기를 초과했습니다.')
        seen = set()
        for e in entries:
            name = e.filename
            rel = PurePosixPath(name)
            if ('\\' in name or ':' in name or rel.is_absolute() or '..' in rel.parts
                    or (e.external_attr >> 16) & 0o170000 == 0o120000
                    or name.casefold() in seen):
                raise ValueError('안전하지 않거나 중복된 ZIP 경로입니다.')
            seen.add(name.casefold())
        z.extractall(destination)
    return inspect_bundle(destination)
