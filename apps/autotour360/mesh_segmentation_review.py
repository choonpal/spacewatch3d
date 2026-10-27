"""Read-only inspection assets from a completed job's original segmented mesh/RGB.

No registration, segmentation or change inference is run here. Projection uses
each capture's original calibration so mismatched boundaries remain visible.
"""
import hashlib
import json
from pathlib import Path
import re
import threading


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def source_file(root, name):
    path = (root / name).resolve()
    if not path.is_relative_to(root.resolve()) or not path.is_file():
        raise ValueError('입력에 포함된 파일만 검사할 수 있습니다.')
    return path


def colour(ids):
    import numpy as np
    ids = np.asarray(ids, dtype='i8')
    rgb = np.stack([(ids*73+43)%190+40, (ids*131+71)%190+40,
                    (ids*191+19)%190+40], axis=-1).astype('u1')
    rgb[ids <= 0] = [155, 162, 172]
    return rgb


def camera_from_world(pose):
    import numpy as np
    pose = np.asarray(pose, float)
    if pose.shape != (4, 4) or not np.isfinite(pose).all():
        raise ValueError('촬영 위치와 방향이 올바르지 않습니다.')
    return (np.diag([1., -1., -1., 1.]) @ np.linalg.inv(pose)).tolist()


def build(inputs, output):
    import numpy as np
    from PIL import Image
    doc = json.loads((inputs/'manifest.json').read_text(encoding='utf-8-sig'))
    output.mkdir(parents=True, exist_ok=True)
    scenes = []
    for epoch, phase in enumerate(['before', 'after']):
        spec = doc[phase]
        if spec.get('projection', 'erp') != 'erp':
            raise ValueError('이 검증 화면은 360도 ERP 원본을 지원합니다.')
        mesh_path = source_file(inputs, spec['mesh'])
        with np.load(mesh_path, allow_pickle=False) as z:
            vertices, faces, ids = z['vertices'], z['faces'], np.maximum(0, z['face_instance_id'].astype('i8'))
        labels = {}
        if spec.get('objects'):
            report = json.loads(source_file(inputs, spec['objects']).read_text(encoding='utf-8-sig'))
            labels = {int(o['instance_id']): o.get('label', 'unknown') for o in report.get('instance_records', [])}
        # Never display a person instance as static segmented geometry.
        ids[np.isin(ids, [i for i, label in labels.items() if label == 'person'])] = 0
        matrix = np.asarray(spec.get('world_from_mesh', np.eye(4)), float)
        world = vertices @ matrix[:3, :3].T + matrix[:3, 3]
        positions = world[faces].reshape(-1, 3).astype('<f4')
        rgba = np.column_stack([colour(ids), np.where(ids > 0, 255, 0).astype('u1')])
        rgba = np.repeat(rgba, 3, axis=0)
        mesh_name = f'mesh-{epoch}.bin'
        (output/mesh_name).write_bytes(positions.tobytes() + rgba.tobytes() + ids.astype('<u4').tobytes())
        objects = [dict(id=int(i), label=labels.get(int(i), 'unknown'), faces=int((ids == i).sum()),
                        colour=colour(np.array([i]))[0].tolist()) for i in np.unique(ids) if i > 0]
        frames = []
        for index, frame in enumerate(spec['frames']):
            if 'image' not in frame:
                raise ValueError('검증에 사용할 원본 추출 프레임이 필요합니다.')
            image_path = source_file(inputs, frame['image'])
            actual_sha = digest(image_path)
            for key in ['sha256', 'segmentation_rgb_sha256']:
                if frame.get(key) and frame[key] != actual_sha:
                    raise ValueError('원본 RGB와 분할 마스크의 연결 해시가 다릅니다.')
            stem = f'{epoch}-{index:05d}'
            with Image.open(image_path) as im:
                rgb = im.convert('RGB'); rgb.thumbnail((3072, 1536), Image.Resampling.LANCZOS)
                rgb.save(output/f'rgb-{stem}.jpg', quality=94)
            person = np.zeros((512, 1024), bool)
            outline = np.zeros((512, 1024, 4), 'u1')
            source_seg_sha = None
            if frame.get('segmentation'):
                path = source_file(inputs, frame['segmentation']); source_seg_sha = digest(path)
                with np.load(path, allow_pickle=False) as z:
                    cls, instance, person = z['class_id'], z['instance_id'], z['person'].astype(bool)
                if cls.ndim != 2 or cls.shape != instance.shape or cls.shape != person.shape:
                    raise ValueError('원본 분할 마스크 크기가 일치하지 않습니다.')
                identity = cls.astype('i8')*2**32 + instance
                edge = identity != np.roll(identity, 1, axis=1)
                edge[1:] |= identity[1:] != identity[:-1]
                valid = (cls > 0) & ~person
                outline = np.zeros((*cls.shape, 4), 'u1')
                outline[..., :3] = [83, 255, 152]
                outline[(instance == 0) & valid, :3] = [255, 110, 211]
                outline[..., 3] = np.where(valid & edge, 255, 0)
            if frame.get('person_mask'):
                with Image.open(source_file(inputs, frame['person_mask'])) as im:
                    person |= np.asarray(im.convert('L').resize(person.shape[::-1], Image.Resampling.NEAREST)) > 0
            Image.fromarray(person.astype('u1')*255).save(output/f'person-{stem}.png')
            Image.fromarray(outline).save(output/f'mask-{stem}.png')
            frames.append(dict(id=str(frame['id']), timestamp_s=frame.get('timestamp_s'),
                image=f'rgb-{stem}.jpg', person=f'person-{stem}.png', outline=f'mask-{stem}.png',
                has_segmentation=source_seg_sha is not None, rgb_sha256=actual_sha,
                segmentation_sha256=source_seg_sha, camera_from_world=camera_from_world(frame['world_from_camera'])))
        scenes.append(dict(epoch=epoch, name=spec['id'], mesh=mesh_name, vertices=len(positions),
            faces=len(faces), labelled_faces=int((ids > 0).sum()), objects=objects, frames=frames,
            source_mesh_sha256=digest(mesh_path), bounds=[positions.min(0).tolist(), positions.max(0).tolist()]))
    result = dict(scenes=scenes, scope=doc.get('scope', ''), inference_rerun=False,
                  projection='Original capture calibration; no alignment or fitted image warp',
                  person_mask_source='Supplied original segmentation / person masks')
    (output/'manifest.json').write_text(json.dumps(result, ensure_ascii=False), encoding='utf-8')
    return result


class ReviewAssets:
    def __init__(self, jobs):
        self.jobs, self.lock = jobs, threading.Lock()

    def asset(self, key, name):
        if not (name == 'manifest.json' or re.fullmatch(r'mesh-[01]\.bin|(?:rgb|person|mask)-[01]-\d{5}\.(?:jpg|png)', name)):
            raise KeyError('검증 화면 파일이 없습니다.')
        folder = self.jobs.folder(key)
        if self.jobs.status(key)['state'] != 'complete':
            raise KeyError('완료된 입력의 분할 결과만 검사할 수 있습니다.')
        inputs = folder/'inputs'
        doc = json.loads((inputs/'manifest.json').read_text(encoding='utf-8-sig'))
        paths = [inputs/'manifest.json']
        for phase in ['before', 'after']:
            spec = doc[phase]
            paths += [source_file(inputs, spec[k]) for k in ['mesh', 'objects'] if k in spec]
            paths += [source_file(inputs, f[k]) for f in spec['frames'] for k in ['image', 'segmentation', 'person_mask'] if k in f]
        # Completed input bundles are immutable. Invalidate inspection exports
        # on source file replacement as well as formatter/version changes.
        signature = digest(__file__) + repr([(str(p), p.stat().st_size, p.stat().st_mtime_ns) for p in paths])
        version = hashlib.sha256(signature.encode()).hexdigest()[:20]
        target = folder/'segmentation-review'/version
        with self.lock:
            if not (target/'manifest.json').is_file():
                build(inputs, target)
        path = target/name
        if not path.is_file():
            raise KeyError('검증 화면 파일이 없습니다.')
        return path
