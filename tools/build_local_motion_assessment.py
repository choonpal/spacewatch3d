"""Attach local alignment-error evidence to existing Rescan object associations.

Writes a separate report. Original meshes, poses, labels and changes stay intact.
"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree
from local_motion_error import CONFIG, local_model, assess_motion


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def pair_background(samples, normals, plane_pairs, rotation, raw_support):
    """Mutual same-plane geometric correspondences, no instance/class labels."""
    source, target, matched_normals, plane_ids = [], [], [], []
    for k, pair in enumerate(plane_pairs):
        selected = []
        for e, plane in enumerate(pair['planes']):
            p = samples[e] @ rotation.T; n = normals[e] @ rotation.T
            plane_normal = np.array(plane['normal']); bounds = plane['bounds']
            mask = (np.abs(p @ plane_normal+plane['offset_m']) <= .025)
            mask &= np.abs(n @ plane_normal) >= np.cos(np.deg2rad(20))
            mask &= ((p >= np.array(bounds['min'])-.025) & (p <= np.array(bounds['max'])+.025)).all(1)
            mask &= raw_support[e] <= .045
            selected.append(samples[e][mask])
        before, after = selected
        if min(len(before), len(after)) < 10:
            continue
        distance, nearest = cKDTree(before).query(after)
        reverse = cKDTree(after).query(before)[1]
        keep = (distance <= .12) & (reverse[nearest] == np.arange(len(after)))
        a, b = after[keep], before[nearest[keep]]
        source.append(a); target.append(b)
        matched_normals.append(np.tile(np.array(pair['normal']) @ rotation, (len(a), 1)))
        plane_ids.extend([k]*len(a))
    if not source:
        return np.empty((0, 3)), np.empty((0, 3)), np.empty((0, 3)), np.array([], int)
    return np.concatenate(source), np.concatenate(target), np.concatenate(matched_normals), np.array(plane_ids)


def outside_object(points, region, rotation, margin):
    display = points @ rotation.T; keep = np.ones(len(points), bool)
    for name in ('before_bounds', 'after_bounds'):
        box = region.get(name)
        if box:
            keep &= ~((display >= np.array(box['min'])-margin) & (display <= np.array(box['max'])+margin)).all(1)
    return keep


def run(review_dir, structural_dir, output, *, expanded=False, baseline_path=None):
    import open3d as o3d
    from compare_mesh_methods import surface_samples
    from plyfile import PlyData
    from stable_surface_patches import PATCH_CONFIG, discover_stable_patches, screen_local_directions, heldout_direction_checks
    if output.exists() and any(output.iterdir()):
        raise ValueError('Use a new empty output directory')
    output.mkdir(parents=True, exist_ok=True)
    (output/'protocol.json').write_text(json.dumps({'motion_config': CONFIG, 'expanded_references': expanded,
        'patch_config': PATCH_CONFIG if expanded else None,
        'reference_inputs': 'mesh XYZ/normals and original XYZ support only; no instance/class/change labels',
        'selection': 'smallest radius with fully observed translation; otherwise most observed dimensions, then smallest radius'}, indent=2), encoding='utf-8')
    manifest = load(review_dir/'changes.json'); previous = load(structural_dir/'comparison.json')
    plane_pairs = load(structural_dir/'plane-extraction.json')['pairs']
    rotation = np.array(manifest['display_rotation'])
    mesh_paths = [review_dir/f'cache/{e}-depth8.ply' for e in (0, 1)]
    mesh_hashes = list(map(digest, mesh_paths))
    if (mesh_hashes != previous['source_hashes'] or previous['source_epochs'] != manifest['source_epochs']
            or not np.allclose(rotation, previous['display_rotation'])):
        raise ValueError('Structural reference belongs to different input meshes/captures')
    samples, normals, support = [], [], []
    for e, path in enumerate(mesh_paths):
        mesh = o3d.io.read_triangle_mesh(str(path)); p, n = surface_samples(mesh, 20261015+e)
        raw_path = Path(manifest['source_files'][e]['path'])
        if digest(raw_path) != manifest['source_files'][e]['sha256']:
            raise ValueError('Original scan changed')
        vertex = PlyData.read(raw_path)['vertex'].data
        raw = np.column_stack([vertex[key] for key in ('x', 'y', 'z')]).astype(float)
        if e:
            t = np.array(manifest['registration']['after_to_before']); raw = raw @ t[:3, :3].T+t[:3, 3]
        samples.append(p); normals.append(n); support.append(cKDTree(raw).query(p)[0])
    a, b, n, planes = pair_background(samples, normals, plane_pairs, rotation, support)
    print('Common structural correspondences:', len(a), flush=True)
    catalog = []
    if expanded:
        supported = [support[e] <= .045 for e in (0, 1)]
        extra_a, extra_b, extra_n, extra_ids, catalog = discover_stable_patches(
            samples[0][supported[0]], samples[1][supported[1]], normals[0][supported[0]], normals[1][supported[1]])
        print('All-orientation common patch candidates:', len(catalog), 'pairs:', len(extra_a), flush=True)
        a = np.concatenate([a, extra_a]); b = np.concatenate([b, extra_b]); n = np.concatenate([n, extra_n])
        planes = np.r_[-1-planes, extra_ids]
        # Preserve original matches first; do not double-weight the same sampled source point.
        keep = np.sort(np.unique(np.round(a, 7), axis=0, return_index=True)[1])
        a, b, n, planes = a[keep], b[keep], n[keep], planes[keep]
        (output/'patch-candidates.json').write_text(json.dumps(catalog, indent=2), encoding='utf-8')
    reports = []; records = []
    for region in manifest['regions']:
        if region['type'] != 'moved':
            reports.append(assess_motion(region, {})); continue
        m = region['motion']; endpoints = np.array([m['before_reference_m'], m['after_reference_m']])
        excluded = outside_object(a, region, rotation, CONFIG['object_exclusion_margin_m']) & outside_object(b, region, rotation, CONFIG['object_exclusion_margin_m'])
        distance = np.min(np.linalg.norm(((a+b)/2)[:, None, :]-endpoints[None, :, :], axis=2), axis=1)
        model = {'usable': False, 'reason': 'insufficient_background'}; chosen = np.zeros(len(a), bool); attempts = []
        best = None
        for radius in CONFIG['radii_m']:
            selected = excluded & (distance <= radius)
            directions = []; validation = []
            if expanded and selected.any():
                indices = np.flatnonzero(selected)
                admitted, directions = screen_local_directions(a[selected], b[selected], n[selected], planes[selected], endpoints.mean(0))
                selected[indices[~admitted]] = False
            attempt = local_model(a[selected], b[selected], n[selected], endpoints.mean(0), endpoints[1])
            if expanded and attempt.get('usable'):
                validation = heldout_direction_checks(a[selected], b[selected], n[selected], planes[selected], endpoints.mean(0), attempt)
                failed = [check['family'] for check in validation if not check['passed']]
                if failed:
                    from local_motion_error import normal_families
                    membership = normal_families(n[selected])[1]; indices = np.flatnonzero(selected)
                    selected[indices[np.isin(membership, failed)]] = False
                    attempt = local_model(a[selected], b[selected], n[selected], endpoints.mean(0), endpoints[1])
            if expanded and attempt.get('reason') == 'background_correction_too_large':
                selected &= planes < 0
                attempt = local_model(a[selected], b[selected], n[selected], endpoints.mean(0), endpoints[1])
                attempt['extra_surface_rejection'] = 'inconsistent_background_correction'
            chosen = selected
            attempt.update(radius_m=radius, plane_ids=sorted(set(int(v) for v in planes[chosen])))
            if expanded:
                patch_ids = sorted(set(int(v) for v in planes[chosen] if v >= 0))
                preview_ids = [patch_ids[i] for i in np.linspace(0, len(patch_ids)-1, min(32, len(patch_ids))).astype(int)] if patch_ids else []
                attempt.update(additional_patch_count=len(patch_ids), additional_pair_count=int((planes[chosen] >= 0).sum()),
                    direction_support=directions, heldout_new_directions=validation,
                    reference_preview=[catalog[i] for i in preview_ids])
            attempts.append({key: attempt.get(key) for key in ('radius_m', 'reason', 'pair_count', 'normal_family_count', 'observable_dimensions')})
            model = attempt
            if expanded:
                rank = attempt.get('observable_dimensions', 0) if attempt.get('usable') else 0
                if best is None or rank > best[0]:
                    best = (rank, attempt, chosen.copy())
                if attempt.get('directions_complete') or attempt.get('reason') == 'background_correction_too_large':
                    break
                continue
            # Expand only when support/observability is insufficient, never because motion was not detected.
            if model.get('reason') == 'background_correction_too_large' or (model.get('usable') and model.get('normal_family_count', 0) >= 2):
                break
        if expanded and best is not None:
            _, model, chosen = best
        model['radius_attempts'] = attempts
        assessment = assess_motion(region, model)
        reports.append(assessment)
        records.append({'region_id': region['id'], 'pair_indices': np.flatnonzero(chosen).tolist()})
        print(region['id'], region['name'], assessment['status'], assessment['reason'],
              'observed', assessment.get('observed_displacement_m'), 'remaining', assessment.get('corrected_observable_displacement_m'),
              'envelope', assessment.get('comparison_envelope_m'), 'directions', model.get('observable_dimensions'), flush=True)
    # Baseline decisions are read only AFTER reference discovery, fitting and classification.
    baseline = load(baseline_path) if baseline_path else None
    if baseline and (baseline['source_changes_sha256'] != digest(review_dir/'changes.json') or baseline['aligned_meshes_sha256'] != mesh_hashes):
        raise ValueError('Baseline belongs to other input data')
    previous_assessments = {row['region_id']: row for row in baseline['assessments']} if baseline else {}
    for row in reports:
        if row['region_id'] in previous_assessments:
            row['baseline'] = previous_assessments[row['region_id']]
    report = {'version': 1, 'kind': 'local_background_motion_assessment', 'source_scene': manifest['scene'],
        'source_epochs': manifest['source_epochs'], 'source_changes_sha256': digest(review_dir/'changes.json'),
        'source_meshes_sha256': [display['mesh']['sha256'] for display in manifest['displays']],
        'aligned_meshes_sha256': mesh_hashes, 'source_registration': manifest['registration']['after_to_before'],
        'config': CONFIG, 'expanded_reference_search': expanded, 'patch_config': PATCH_CONFIG if expanded else None,
        'baseline_report_sha256': digest(baseline_path) if baseline_path else None,
        'reference_selection_uses_labels': False, 'assessments': reports,
        'summary': {key: sum(r['status'] == key for r in reports) for key in ('motion_supported', 'alignment_explained', 'uncertain', 'not_applicable')},
        'limitations': ['Common structural surfaces are assumed stable; the model does not prove that assumption.',
            'Near-parallel normals are grouped; unobservable displacement directions are not reported as measured.',
            'Spatial holdout and block resampling bound local residuals empirically; this is not calibrated statistical confidence.',
            'Existing Rescan object associations and rigid-motion estimates are inputs; this does not make detection label-free.',
            'Camera/image-projection errors are not estimated by this geometry-only motion assessment.',
            'The analytical correction does not modify meshes, photos, or camera poses.']}
    (output/'motion-assessment.json').write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    (output/'support-audit.json').write_text(json.dumps(records, indent=2), encoding='utf-8')
    np.savez_compressed(output/'background-correspondences.npz', source=a, target=b, normals=n, plane_ids=planes)
    print('SUMMARY', report['summary'], flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--review-dir', type=Path, required=True)
    parser.add_argument('--structural-dir', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--expanded-references', action='store_true')
    parser.add_argument('--baseline-assessment', type=Path)
    args = parser.parse_args()
    run(args.review_dir, args.structural_dir, args.output, expanded=args.expanded_references, baseline_path=args.baseline_assessment)
