"""Bounded reader for standard, uncompressed 3DGS PLY (never pickle/checkpoints)."""
import hashlib
import json
import math
import struct
from pathlib import Path

TYPES = {'float': ('f', 4), 'float32': ('f', 4), 'double': ('d', 8), 'float64': ('d', 8),
         'uchar': ('B', 1), 'uint8': ('B', 1), 'int': ('i', 4), 'uint': ('I', 4)}
REQUIRED = {'x', 'y', 'z', 'opacity', *[f'scale_{i}' for i in range(3)],
            *[f'rot_{i}' for i in range(4)], *[f'f_dc_{i}' for i in range(3)]}


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(4 * 1024**2), b''):
            h.update(block)
    return h.hexdigest()


def header(path):
    with Path(path).open('rb') as f:
        if f.readline(16) != b'ply\n':
            f.seek(0)
            if f.readline(16).strip() != b'ply':
                raise ValueError('3DGS PLY 파일이 아닙니다.')
        fields, count, fmt, element = [], None, None, None
        while f.tell() < 65536:
            line = f.readline(65537-f.tell())
            if not line:
                raise ValueError('PLY 헤더가 잘렸습니다.')
            words = line.decode('ascii').strip().split()
            if not words or words[0] in ('comment', 'obj_info'):
                continue
            if words[0] == 'end_header':
                break
            if words[0] == 'format':
                if len(words)!=3:raise ValueError('PLY format 헤더가 올바르지 않습니다.')
                fmt = words[1]
            elif words[0] == 'element':
                if len(words)!=3:raise ValueError('PLY element 헤더가 올바르지 않습니다.')
                element = words[1]
                if element != 'vertex':
                    raise ValueError('객체 메쉬가 아닌 표준 Gaussian PLY를 선택해 주세요.')
                count = int(words[2])
            elif words[0] == 'property':
                if element != 'vertex' or len(words) != 3 or words[1] not in TYPES:
                    raise ValueError('지원하지 않는 PLY 속성 형식입니다.')
                fields.append((words[2], words[1]))
        else:
            raise ValueError('PLY 헤더가 너무 큽니다.')
        if fmt != 'binary_little_endian':
            raise ValueError('binary_little_endian 형식의 원본 3DGS PLY를 사용해 주세요.')
        names = [n for n, _ in fields]
        if len(names) != len(set(names)) or not REQUIRED.issubset(names) or len(fields) > 96:
            raise ValueError('Gaussian 위치·SH 색상·opacity·scale·rotation 속성이 필요합니다.')
        rest = sorted(int(n[7:]) for n in names if n.startswith('f_rest_'))
        if rest not in [list(range(n)) for n in (0, 9, 24, 45)]:
            raise ValueError('SH 차수 0~3의 표준 Gaussian PLY를 사용해 주세요.')
        if count is None or not 100 <= count <= 10_000_000:
            raise ValueError('Gaussian 개수는 100~10,000,000개여야 합니다.')
        row_bytes = sum(TYPES[t][1] for _, t in fields)
        offset = f.tell()
        if Path(path).stat().st_size != offset + count * row_bytes:
            raise ValueError('PLY 크기가 헤더와 다릅니다. 파일 전체를 업로드해 주세요.')
        fmt_row = '<' + ''.join(TYPES[t][0] for _, t in fields)
        for _ in range(min(count, 64)):
            values = struct.unpack(fmt_row, f.read(row_bytes))
            if not all(math.isfinite(v) for v in values):
                raise ValueError('Gaussian에 유효하지 않은 숫자가 있습니다.')
        return {'count': count, 'fields': fields, 'offset': offset, 'row_bytes': row_bytes,
                'sh_degree': int(math.sqrt(len(rest) // 3 + 1)) - 1}


def load(path):
    import numpy as np
    h = header(path)
    dtype = np.dtype([(n, '<' + TYPES[t][0]) for n, t in h['fields']])
    raw = np.memmap(path, dtype=dtype, mode='r', offset=h['offset'], shape=(h['count'],))
    stack = lambda names: np.column_stack([raw[n] for n in names]).astype('f4')
    xyz = stack(['x', 'y', 'z'])
    dc = stack([f'f_dc_{i}' for i in range(3)])[:, None, :]
    rest_count = 3 * ((h['sh_degree'] + 1) ** 2 - 1)
    rest = stack([f'f_rest_{i}' for i in range(rest_count)]).reshape(len(xyz), 3, -1).transpose(0, 2, 1) if rest_count else np.empty((len(xyz), 0, 3), 'f4')
    result = {'xyz': xyz, 'sh': np.concatenate([dc, rest], 1),
              'scale': stack([f'scale_{i}' for i in range(3)]),
              'quat': stack([f'rot_{i}' for i in range(4)]), 'opacity': raw['opacity'].astype('f4'),
              'degree': h['sh_degree']}
    for key in ['xyz', 'sh', 'scale', 'quat', 'opacity']:
        if not np.isfinite(result[key]).all():
            raise ValueError('Gaussian 전체 검사에서 유효하지 않은 숫자를 찾았습니다.')
    if np.any(np.linalg.norm(result['quat'], axis=1) < 1e-8) or np.abs(result['scale']).max() > 40:
        raise ValueError('Gaussian rotation 또는 scale 범위가 올바르지 않습니다.')
    return result


def camera_document(path):
    d = json.loads(Path(path).read_text(encoding='utf-8-sig'))
    if not isinstance(d, dict) or not any(k in d for k in ('before', 'after')):
        raise ValueError('카메라 JSON에는 before 또는 after 시점 목록이 필요합니다.')
    for phase in ('before', 'after'):
        rows = d.get(phase, [])
        if not isinstance(rows, list) or len(rows) > 2000:
            raise ValueError('각 회차의 카메라 목록은 최대 2,000개입니다.')
        for row in rows:
            m = row.get('camera_to_world') if isinstance(row, dict) else None
            if not isinstance(m, list) or len(m) != 4 or any(not isinstance(r, list) or len(r) != 4 for r in m):
                raise ValueError('카메라는 원본 맵 좌표의 4×4 camera_to_world(CV) 행렬이어야 합니다.')
            def finite(value):
                try:return isinstance(value,(int,float)) and math.isfinite(value)
                except OverflowError:return False
            if any(not finite(v) for r in m for v in r):
                raise ValueError('카메라 행렬의 숫자가 올바르지 않습니다.')
            if any(abs(m[3][i] - [0, 0, 0, 1][i]) > 1e-6 for i in range(4)):
                raise ValueError('카메라 동차 행렬의 마지막 행이 올바르지 않습니다.')
    return d


def write_subset(source, destination, mask):
    """Preserve all original Gaussian attributes when partitioning the display map."""
    import numpy as np
    h = header(source)
    dtype = np.dtype([(n, '<' + TYPES[t][0]) for n, t in h['fields']])
    raw = np.memmap(source, dtype=dtype, mode='r', offset=h['offset'], shape=(h['count'],))
    count = int(np.count_nonzero(mask))
    text = 'ply\nformat binary_little_endian 1.0\nelement vertex ' + str(count) + '\n'
    text += ''.join('property ' + typ + ' ' + name + '\n' for name,typ in h['fields']) + 'end_header\n'
    with Path(destination).open('wb') as f:
        f.write(text.encode('ascii'));f.write(raw[mask].tobytes())
    return count
