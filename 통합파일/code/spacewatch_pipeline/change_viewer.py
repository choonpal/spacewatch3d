"""Publish the exact surface patches in the detector's baseline coordinate frame."""
from pathlib import Path
import numpy as np
from .geometry import similarity, transform
from .io import read, save, sha


def publish_surface_comparison(tour, job, bundle, trusted, assets, mesh_writer):
    """No detection, registration, face reordering, or object matching occurs here."""
    alignment = tour['alignment']
    status = alignment['status']
    available = status != 'unavailable'
    matrix = alignment.get('baseline_from_current')
    if available:
        matrix, scale = similarity(matrix)
        if not np.isclose(scale, 1):
            raise ValueError('Surface comparison requires the detector SE3 transform')
    elif matrix is not None or tour.get('events'):
        raise ValueError('Unavailable comparison cannot contain a transform or patches')
    captures = job['config']['captures']
    if len(captures) != 2:
        raise ValueError('Surface comparison requires exactly two captures')
    maps, events = [], []
    root = Path(bundle) / 'surface-comparison'
    root.mkdir(exist_ok=True)
    if available:
        for epoch, capture in enumerate(captures):
            name = capture['id']
            seg = job['dependencies'][name + '.semantic']
            source = trusted(seg['arrays'])
            with np.load(source) as data:
                vertices, faces, rgb, labels = [data[k] for k in ['vertices', 'faces', 'rgb', 'face_class_id']]
            vertices = transform(vertices, np.eye(4) if epoch == 0 else matrix)
            catalog = read(trusted(seg['vocabulary']))
            filename = f'surface-comparison/mesh-{epoch}.bin'
            mesh_writer(vertices, faces, rgb, labels, catalog, Path(bundle) / filename)
            assets.append(filename)
            maps.append(dict(id=name, epoch=epoch, mesh='/pipeline-assets/' + filename,
                             faces=len(faces), vertices=len(vertices),
                             bounds=[vertices.min(0).tolist(), vertices.max(0).tolist()],
                             source_arrays_sha256=sha(source)))
            used = set()
            for event in tour.get('events', []):
                if event['epoch'] != epoch:
                    continue
                indices = np.asarray(event['face_ids'])
                if (indices.ndim != 1 or indices.dtype.kind not in 'iu' or not len(indices)
                        or indices.min() < 0 or indices.max() >= len(faces)
                        or len(np.unique(indices)) != len(indices) or used.intersection(indices.tolist())):
                    raise ValueError('Invalid or overlapping surface face IDs')
                used.update(indices.tolist())
                points = vertices[np.unique(faces[indices])]
                bounds = np.array([points.min(0), points.max(0)])
                if not np.allclose(bounds, event['bounds'], atol=1e-5, rtol=1e-6):
                    raise ValueError('Surface patch bounds do not match its source mesh and transform')
                if event['status'] not in ['candidate', 'deferred']:
                    raise ValueError('Unknown surface decision')
                # Keep the actual detector evidence, but do not publish server image paths.
                row = {k: v for k, v in event.items() if k not in ['own_views', 'opposite_views']}
                for field in ['own_views', 'opposite_views']:
                    row[field] = [{k: v for k, v in view.items() if k != 'source_image'}
                                  for view in event.get(field, [])]
                events.append(row)
        if len(events) != len(tour.get('events', [])) or len({e['id'] for e in events}) != len(events):
            raise ValueError('Invalid surface epoch or duplicate event ID')
    scene = read(trusted(job['dependencies'][captures[0]['id'] + '.scene']['scene']))
    bounds = None if not maps else [np.min([m['bounds'][0] for m in maps], axis=0).tolist(),
                                   np.max([m['bounds'][1] for m in maps], axis=0).tolist()]
    data = dict(schema_version=1, scope='surface_regions_not_object_events',
                run_id=Path(job['run_root']).name, available=available,
                baseline_capture=captures[0]['id'], current_capture=captures[1]['id'],
                alignment={k: v for k, v in alignment.items() if k != 'alignment'},
                coordinate_frame=captures[0]['id'], bounds=bounds,
                floor_height=scene['floor']['height_m'], maps=maps, events=events,
                audit=tour.get('audit', []), no_object_identity_claim=True,
                metric_scale_verified=all(job['dependencies'][c['id'] + '.reconstruct']['metric_scale_verified'] for c in captures),
                counts=dict(candidates=sum(e['status'] == 'candidate' for e in events),
                            deferred=sum(e['status'] == 'deferred' for e in events)),
                source_changes_sha256=sha(trusted(job['dependencies']['pair.changes']['changes'])))
    save(root / 'comparison.json', data)
    assets.append('surface-comparison/comparison.json')
    return '/pipeline-assets/surface-comparison/comparison.json'
