"""Small, CPU-only provenance and panorama helpers."""
import hashlib
import json
import re
from pathlib import Path
import numpy as np


def read(path):
    return json.loads(Path(path).read_text())


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.partial')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    temp.replace(path)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b''):
            h.update(block)
    return h.hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


VIEWS = [dict(name=n, yaw=y, pitch=p) for n, y, p in
         [('front', 0, 0), ('right', 90, 0), ('back', 180, 0),
          ('left', -90, 0), ('up', 0, 90), ('down', 0, -90)]]


def basis(spec):
    yaw, pitch = np.deg2rad([spec['yaw'], spec['pitch']])
    forward = np.array([np.cos(pitch)*np.sin(yaw), np.sin(pitch), np.cos(pitch)*np.cos(yaw)])
    right = np.array([np.cos(yaw), 0, -np.sin(yaw)])
    return forward, right, np.cross(right, forward)


def perspective_map(width, height, spec, size=512, fov=100):
    y, x = np.mgrid[:size, :size]
    scale = np.tan(np.deg2rad(fov/2))
    forward, right, down = basis(spec)
    d = forward + (2*(x[..., None]+.5)/size-1)*scale*right + (2*(y[..., None]+.5)/size-1)*scale*down
    d /= np.linalg.norm(d, axis=-1, keepdims=True)
    mx = ((np.arctan2(d[..., 0], d[..., 2])/(2*np.pi)+.5)*width-.5).astype('f4')
    my = ((.5-np.arcsin(d[..., 1])/np.pi)*height-.5).astype('f4')
    return mx, np.clip(my, 0, height-1)


def erp_to_view(shape, spec, size=512, fov=100):
    h, w = shape
    y, x = np.mgrid[:h, :w]
    lon, lat = ((x+.5)/w-.5)*2*np.pi, (.5-(y+.5)/h)*np.pi
    d = np.stack([np.cos(lat)*np.sin(lon), np.sin(lat), np.cos(lat)*np.cos(lon)], -1)
    forward, right, down = basis(spec)
    z = d @ forward
    scale = np.tan(np.deg2rad(fov/2))
    px = ((d @ right)/np.maximum(z, 1e-10)/scale+1)*size/2-.5
    py = ((d @ down)/np.maximum(z, 1e-10)/scale+1)*size/2-.5
    valid = (z>0) & (px>=-.5) & (px<size-.5) & (py>=-.5) & (py<size-.5)
    ix = np.clip(np.rint(px), 0, size-1).astype(int)
    iy = np.clip(np.rint(py), 0, size-1).astype(int)
    return valid.ravel(), (iy*size+ix).ravel()


# These are linguistic aliases, never a list of classes to detect.
ALIASES = {'chairs': 'chair', 'seats': 'chair', 'seat': 'chair', 'people': 'person',
           'man': 'person', 'woman': 'person', 'men': 'person', 'women': 'person',
           'couch': 'sofa', 'couches': 'sofa', 'television': 'tv', 'trash bin': 'trash can',
           'garbage can': 'trash can', 'light fixture': 'ceiling light', 'tile floor':'floor',
           'dining table':'table', 'potted plant':'plant', 'hair':'person', 'human face':'person',
           'face':'person', 'head':'person', 'human hand':'person', 'footwear':'person', 'glasses':'glasses'}
ADJECTIVES = {'black','white','red','blue','green','yellow','gray','grey','brown','orange',
              'pink','purple','beige','silver','metal','wooden','plastic','small','large',
              'big','long','short','tall','round','rectangular','square','empty','open','closed','dark','bright',
              'shiny','textured','smooth','thin','thick','single','double'}
STRUCTURE = {'wall','floor','ceiling','carpet','tactile paving','ground','hallway','room','building','sidewalk','tiled wall'}
DYNAMIC = {'person','dog','cat','bird','car','bus','bicycle','motorcycle'}


def canonicalize(text):
    text = re.sub(r'[^a-z -]', ' ', text.lower()).replace('-', ' ')
    if re.search(r'\b(person|man|woman|human)\b',text):return 'person'
    text = re.sub(r'\b(a|an|the|several|some|many|two|three|four|five|six|pair|set|row|rows|group|of)\b', ' ', text)
    text = re.split(r'\b(on|in|with|near|next|behind|under|above|against|that|which|at|and|to)\b', text)[0]
    words = [w for w in text.split() if w not in ADJECTIVES]
    if not words or len(words)>5:
        return None
    text = ' '.join(words)
    if text in ALIASES:
        return ALIASES[text]
    # Inflection is handled by WordNet in normalize.py; suffix stripping damages
    # valid nouns such as gas, glasses and compass.
    text = ' '.join(words)
    return ALIASES.get(text, text)


def catalog_for(names, evidence=None):
    import colorsys
    names = sorted(set(names))
    if not 2 <= len(names) <= 254:
        raise ValueError('OOOPS requires 2..254 discovered/reserved classes')
    return [dict(id=i, name=name, name_ko=name,
                 rgb=[round(v*255) for v in colorsys.hsv_to_rgb((i*.61803398875)%1,.65,.9)],
                 static_mesh=name not in DYNAMIC, instance_eligible=name not in STRUCTURE|DYNAMIC,
                 evidence=(evidence or {}).get(name, [])) for i,name in enumerate(names)]
