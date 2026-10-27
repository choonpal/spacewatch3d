"""Build the event-focused added/moved/removed viewer from automatic event decisions."""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import shutil

import numpy as np
from PIL import Image, ImageDraw

from .core import cube_rotations, erp_project, intrinsics, rays, read, sha, write, project_in_view
from .regional import aligned_mesh


def silhouette_triangles(points, regions, view, output_size=768, erp=False):
    """Unclamped camera projection: outside vertices must not move to the border."""
    uv = np.zeros((len(points), 2), float)
    depth = np.zeros(len(points), float)
    for region in np.unique(regions):
        chosen = regions == region
        pose = view['region_poses'][region] if 'region_poses' in view else view['pose']
        camera = (points[chosen]-pose[:3, 3]) @ pose[:3, :3]
        pixel = camera @ view['K'].T
        depth[chosen] = camera[:, 2]
        uv[chosen] = pixel[:, :2]/np.maximum(camera[:, 2:3], 1e-8)
    h, w = view['depth'].shape
    if erp:
        uv *= np.array([(output_size-1)/(w-1), (output_size-1)/(h-1)])
    else:
        uv = (uv+.5)*np.array([output_size/w, output_size/h])-.5
    tri = uv.reshape(-1, 3, 2)
    good = (depth.reshape(-1, 3) > 1e-6).all(1) & np.isfinite(tri).all((1, 2))
    good &= (tri.max(1) >= 0).all(1) & (tri.min(1) < output_size).all(1)
    return tri[good]


def build(root, manifest_path):
    root, manifest_path = Path(root), Path(manifest_path).resolve()
    original, prepared = read(manifest_path), read(root / "manifest.json")
    result, appearance = read(root / "results/events.json"), read(root / "results/object-appearance.json")
    scene = read(root / "preview/scene.json")
    photo_dir = root / "events/photos"
    photo_dir.mkdir(parents=True, exist_ok=True)
    views = {v["id"]: v for v in prepared["views"]}
    captures = [original["before"], original["after"]]
    raw_frames = [{str(f["id"]): f for f in s["frames"]} for s in captures]
    exported = {}
    meshes = [aligned_mesh(root, prepared, spec, epoch, manifest_path.parent)
              for epoch, spec in enumerate(captures)]
    ghost_cache = {}
    panoramas = {}

    def panorama(record, keys):
        meta = views[record['view']]
        if captures[meta['epoch']].get('projection', 'erp') != 'erp':
            return None
        with np.load(root/'views'/(meta['id']+'.npz'), allow_pickle=False) as data:
            target = dict(data)
        rotation = cube_rotations()[meta['face']]
        center = np.array([(record['bbox'][0]+record['bbox'][2])/2,
                           (record['bbox'][1]+record['bbox'][3])/2, 1.])
        center[:2] *= prepared['settings']['size']/768
        direction = (center @ np.linalg.inv(target['K']).T) @ rotation.T
        overlays = []
        for key in keys:
            obj = result['objects'][key]
            vertices, mesh = meshes[obj['epoch']]
            faces = mesh['faces'][obj['original_face_ids']]
            points = vertices[faces].reshape(-1, 3)
            regions = mesh.get('vertex_region', np.ones(len(vertices), int))[faces].reshape(-1)
            camera = np.empty_like(points)
            for region in np.unique(regions):
                chosen = regions == region
                pose = target['region_poses'][region] if 'region_poses' in target else target['pose']
                camera[chosen] = ((points[chosen]-pose[:3, 3]) @ pose[:3, :3]) @ rotation.T
            overlays.append(dict(key=key, epoch=obj['epoch'], positions=np.round(camera*[1,-1,-1], 6).ravel().tolist()))
        return {**panoramas[(meta['epoch'], meta['frame_id'])],
                'direction': (direction*[1,-1,-1]).tolist(), 'ghosts': overlays}

    def ghost(record, key):
        """Project the original object triangles into this exact RGB camera.

        This is a coloured mesh silhouette, not invented texture or an RGB
        input to SimSaC. The counterpart retains its common-map position.
        """
        if not record or not key:
            return None
        cache_key = (record['view'], key)
        if cache_key in ghost_cache:
            return ghost_cache[cache_key]
        obj = result['objects'][key]
        vertices, mesh = meshes[obj['epoch']]
        faces = mesh['faces'][obj['original_face_ids']]
        with np.load(root/'views'/(record['view']+'.npz'), allow_pickle=False) as data:
            target = dict(data)
        points = vertices[faces].reshape(-1, 3)
        regions = mesh.get('vertex_region', np.ones(len(vertices), int))[faces].reshape(-1)
        triangles = silhouette_triangles(points, regions, target, erp=views[record['view']]['face'] != 'perspective')
        overlay = Image.new('RGBA', (768, 768))
        draw = ImageDraw.Draw(overlay)
        colour = (87, 216, 221, 160) if obj['epoch'] == 0 else (255, 194, 106, 160)
        for tri in triangles:
            draw.polygon([tuple(p) for p in tri], fill=colour)
        # A person remains unobservable even when the mesh behind them exists.
        person = np.load(root/'results/views'/(record['view']+'.npz'), allow_pickle=False)['person']
        mask = np.asarray(Image.fromarray(person).resize((768, 768), Image.Resampling.NEAREST))
        rgba = np.array(overlay); rgba[mask, 3] = 0
        name = f'ghost-{record["view"]}-{key}.png'
        Image.fromarray(rgba).save(photo_dir/name)
        y, x = np.nonzero(rgba[..., 3])
        bounds = [int(x.min()), int(y.min()), int(x.max()+1), int(y.max()+1)] if len(x) else None
        value = dict(image='events/photos/'+name, epoch=obj['epoch'], bbox=bounds, kind='projected_mesh_silhouette')
        ghost_cache[cache_key] = value
        return value

    def photograph(record):
        if record is None:
            return None
        vid, meta = record["view"], views[record["view"]]
        epoch = meta["epoch"]
        if vid not in exported:
            spec = captures[epoch]
            frame = raw_frames[epoch][meta["frame_id"]]
            def resolve(value):
                path = Path(value)
                return path if path.is_absolute() else manifest_path.parent / path
            if "image" in frame:
                image_path = resolve(frame["image"])
                if frame.get("sha256") and sha(image_path) != frame["sha256"]:
                    raise ValueError("Evidence image no longer matches original capture")
                image = np.asarray(Image.open(image_path).convert("RGB"))
            else:
                import cv2
                video = cv2.VideoCapture(str(resolve(spec["video"])))
                try:
                    video.set(cv2.CAP_PROP_POS_FRAMES, frame["source_frame_index"])
                    ok, bgr = video.read()
                    if not ok:
                        raise ValueError("Cannot decode original evidence video frame")
                    image = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                finally:
                    video.release()
            if spec.get("projection", "erp") == "erp":
                pano_key = (epoch, meta['frame_id'])
                if pano_key not in panoramas:
                    name = f'pano-E{epoch}-'+hashlib.sha256(str(meta['frame_id']).encode()).hexdigest()[:12]
                    full = Image.fromarray(image)
                    full.thumbnail((3072, 1536), Image.Resampling.LANCZOS)
                    full.save(photo_dir/(name+'.jpg'), quality=94)
                    # Combine the six independently detected person masks in
                    # the original ERP frame so rotating cannot reveal a ghost
                    # through a photographer excluded from comparison.
                    h, w = 512, 1024
                    yy, xx = np.mgrid[:h, :w]
                    lon=(xx+.5)/w*2*np.pi-np.pi;lat=(yy+.5)/h*np.pi-np.pi/2
                    direction=np.stack([np.sin(lon)*np.cos(lat),np.sin(lat),np.cos(lon)*np.cos(lat)],-1)
                    mask=np.zeros((h,w),bool)
                    for other in views.values():
                        if other['epoch'] != epoch or other['frame_id'] != meta['frame_id']:
                            continue
                        r=cube_rotations()[other['face']];camera=direction@r
                        K=intrinsics(prepared['settings']['size'], prepared['settings']['size'])
                        pixel=camera@K.T; uv=np.rint(pixel[...,:2]/np.maximum(camera[...,2:],1e-8)).astype(int)
                        valid=(camera[...,2]>0)&(uv[...,0]>=0)&(uv[...,0]<prepared['settings']['size'])&(uv[...,1]>=0)&(uv[...,1]<prepared['settings']['size'])
                        person=np.load(root/'results/views'/(other['id']+'.npz'))['person']
                        mask[valid] |= person[uv[...,1][valid],uv[...,0][valid]]
                    Image.fromarray(mask.astype('u1')*255).save(photo_dir/(name+'-person.png'))
                    panoramas[pano_key]=dict(image=f'events/photos/{name}.jpg', person_mask=f'events/photos/{name}-person.png',
                                             width=full.width,height=full.height,projection='erp')
                    if 'segmentation' in frame:
                        with np.load(resolve(frame['segmentation']),allow_pickle=False) as seg:
                            ids=seg['class_id'].astype('i8')*100000+seg['instance_id'];valid=(seg['class_id']>0)&~seg['person']
                        edge=np.zeros(ids.shape,bool)
                        for axis in [0,1]:
                            for shift in [-1,1]:edge |= ids != np.roll(ids,shift,axis=axis)
                        pixels=np.where(valid,np.where(edge,255,28),0).astype('u1')
                        Image.fromarray(pixels).save(photo_dir/(name+'-segmentation.png'))
                        panoramas[pano_key]['segmentation']=f'events/photos/{name}-segmentation.png'
                K = intrinsics(768, 768)
                image = erp_project(image, rays(K, (768, 768)), cube_rotations()[meta["face"]])
            else:
                image = np.asarray(Image.fromarray(image).resize((768, 768), Image.Resampling.BILINEAR))
            target = photo_dir / (vid + ".jpg")
            Image.fromarray(image).save(target, quality=94)
            exported[vid] = dict(image=f'events/photos/{vid}.jpg', frame_id=meta["frame_id"],
                    capture_id=meta["capture_id"], timestamp_s=meta["timestamp_s"], face=meta["face"],
                    source_frame_index=meta["source_frame_index"], width=768, height=768,
                    source_inference_view=vid, image_sha256=sha(target))
        scale = 768 / prepared["settings"]["size"]
        return {**exported[vid], **record, "bbox": (np.array(record["bbox"], float)*scale).tolist()}

    events = []
    for event in result["events"]:
        choices = {}
        for epoch in [0, 1]:
            key = event["before"] if epoch == 0 else event["after"]
            if key:
                group = appearance["objects"].get(key, {}).get("observations", [])
                group = sorted(group, key=lambda r: (views[r['view']]['face'] in ['up','down'], -r["pixels"], r["view"]))
                group = [dict(**r, role="observed_object", object_id=key) for r in group]
            else:
                own_key = event["object_ids"][0]
                group = event["vacancy"].get(own_key, [])
                # Preserve usable observations first; mark each as an expected location.
                group = sorted(group, key=lambda r: (-r["observed_pixels"], r["view"]))
            selected, used = [], set()
            for record in group:
                if record["frame_id"] in used:
                    continue
                used.add(record["frame_id"])
                selected.append(photograph(record))
                if len(selected) == 3:
                    break
            choices[str(epoch)] = selected
        photo_sets = []
        for index in range(max([len(v) for v in choices.values()]+[1])):
            photo_sets.append({k: group[min(index, len(group)-1)] if group else None for k, group in choices.items()})
        for pair in photo_sets:
            for epoch, record in pair.items():
                if record:
                    pair[epoch] = {**record, 'ghosts': [ghost(record, key) for key in event['object_ids']],
                                   'panorama': panorama(record, event['object_ids'])}
        events.append({**event, "photo_sets": photo_sets})
    scenes = [{**s, "mesh": "preview/"+s["mesh"], "cloud": "preview/"+s["cloud"]} for s in scene["scenes"]]
    write(root / "viewer-data.json", {**result, "events": events, "scenes": scenes,
          "inference_pairs": read(root / "validation.json")["fresh_inference_pairs"],
          "source_event_sha256": sha(root / "results/events.json")})
    old = root / "index.html"
    if old.exists() and not (root / "raw-report.html").exists():
        shutil.copyfile(old, root / "raw-report.html")
    for source, target in [("events-viewer.html", "index.html"), ("events-viewer.css", "events-viewer.css"),
                           ("events-viewer.js", "events-viewer.js"), ("photo-presentation.js", "photo-presentation.js")]:
        shutil.copyfile(Path(__file__).with_name(source), root / target)
    return root / "index.html"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    print(build(args.root, args.manifest))
