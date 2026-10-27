"""Infer object events from fresh SimSaC masks, source meshes and observed free space."""
from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial import cKDTree

from .core import project, project_in_view, read, sha, transform, write
from .regional import aligned_mesh
from .uncertainty import object_uncertainty


POLICY = dict(min_frames=2, min_shape_f1=.75, min_shape_direction=.60,
              min_appearance=.80, min_match_margin=.06, min_stationary_f1=.55,
              min_free_fraction=.65, max_occluded_fraction=.20,
              min_projected_pixels=12, min_projected_fraction=.20,
              minimum_identity_frame_support=2, object_scale_fitted=False,
              yaw_seeds_degrees=list(range(0, 360, 45)), manual_overrides_used=False)


def identity_score(a, b):
    if a is None or b is None or not len(a) or not len(b):
        return 0.
    cosine = np.clip(np.asarray(a) @ np.asarray(b).T, -1, 1)
    return float((np.median(cosine.max(1)) + np.median(cosine.max(0))) / 2)


def coverage(a, b, tolerance):
    distances = [cKDTree(b).query(a)[0], cKDTree(a).query(b)[0]]
    directional = [float((d <= tolerance).mean()) for d in distances]
    f1 = 2 * np.prod(directional) / max(sum(directional), 1e-12)
    return float(f1), directional


def rigid_fit(a, b, tolerance):
    tree, best = cKDTree(b), None
    for degrees in POLICY["yaw_seeds_degrees"]:
        angle = np.deg2rad(degrees)
        rotation = np.array([[np.cos(angle), 0, np.sin(angle)], [0, 1, 0], [-np.sin(angle), 0, np.cos(angle)]])
        m = np.eye(4)
        m[:3, :3] = rotation
        m[:3, 3] = np.median(b, 0) - rotation @ np.median(a, 0)
        for _ in range(20):
            moved = transform(a, m)
            distance, index = tree.query(moved)
            keep = distance <= np.quantile(distance, .85)
            x, y = moved[keep], b[index[keep]]
            cx, cy = x.mean(0), y.mean(0)
            xx, yy = (x-cx)[:, [0, 2]], (y-cy)[:, [0, 2]]
            angle = np.arctan2(np.sum(xx[:, 1]*yy[:, 0]-xx[:, 0]*yy[:, 1]), np.sum(xx*yy))
            r = np.array([[np.cos(angle), 0, np.sin(angle)], [0, 1, 0], [-np.sin(angle), 0, np.cos(angle)]])
            delta = np.eye(4)
            delta[:3, :3], delta[:3, 3] = r, cy-r@cx
            m = delta @ m
            if np.linalg.norm(delta[:3, 3]) < 1e-5 and abs(angle) < 1e-5:
                break
        moved = transform(a, m)
        f1, directional = coverage(moved, b, tolerance)
        residual = np.r_[tree.query(moved)[0], cKDTree(moved).query(b)[0]]
        item = dict(shape_f1=f1, directional_coverage=directional,
                    median_residual_u=float(np.median(residual)), before_to_after_rigid=m.tolist())
        if best is None or (f1, -item["median_residual_u"]) > (best["shape_f1"], -best["median_residual_u"]):
            best = item
    return best


def free_frames(records):
    unique = {}
    for record in records:
        key = record["frame_id"]
        if key not in unique or record.get("observed_pixels", 0) > unique[key].get("observed_pixels", 0):
            unique[key] = record
    return [r for r in unique.values() if r["usable"] and r["free_fraction"] >= POLICY["min_free_fraction"]
            and r["occluded_fraction"] <= POLICY["max_occluded_fraction"]]


def segmented_background_frames(records):
    return sorted({r['frame_id'] for r in records if r['usable']
                   and r.get('segmentation_available', False)
                   and r.get('unknown_fraction', 1) <= .15
                   and r['occluded_fraction'] <= .05
                   and r.get('same_class_fraction', 1) <= .02
                   and r['occupied_fraction'] + r['free_fraction'] >= .85})


def decide_unmatched(epoch, has_change, own_frames, opposite, plausible_match, own_segmented_frames=0):
    if not has_change:
        return "unchanged", "no_repeatable_simsac_change"
    if own_frames < POLICY["min_frames"]:
        return "unresolved", "own_observation_insufficient"
    if sum(r.get('same_class_fraction', 0) >= .35 and r['usable'] for r in opposite) >= 2:
        return 'unresolved', 'counterpart_in_rgb_segmentation_but_mesh_incomplete'
    if plausible_match:
        return "unresolved", "counterpart_identity_ambiguous"
    # A coarse mesh may reconstruct the wall/floor behind a thin object.
    # Occupied geometry is not proof of that object's presence. Require
    # independent original-image masks on both sides for this separate path.
    background = segmented_background_frames(opposite)
    if own_segmented_frames >= POLICY['min_frames'] and len(background) >= POLICY['min_frames']:
        return ('removed' if epoch == 0 else 'added'), 'segmented_object_appears_over_observed_background'
    if sum(r["usable"] and r["occupied_fraction"] >= .45 for r in opposite) >= 2:
        return "unresolved", "surface_still_occupies_expected_location"
    if len(free_frames(opposite)) >= POLICY["min_frames"]:
        return ("removed" if epoch == 0 else "added"), "expected_location_observed_free_in_independent_frames"
    return "unresolved", "opposite_location_occluded_or_unobserved"


def movement_rejections(pair, origin, destination, error, voxel):
    reasons = []
    for good, reason in [
        (pair.get('alignment_supported', True), 'registration_uncertainty_unsupported'),
        (pair["shape_f1"] >= POLICY["min_shape_f1"], "shape_fit_weak"),
        (min(pair["directional_coverage"]) >= POLICY["min_shape_direction"], "partial_shape_coverage"),
        (pair["appearance"] >= POLICY["min_appearance"], "rgb_identity_weak"),
        (pair["frame_support"] >= POLICY["minimum_identity_frame_support"], "identity_frame_support"),
        (pair["margin"] >= POLICY["min_match_margin"], "repeated_object_identity_ambiguous"),
        (pair["displacement_u"] > max(3*error, 3*voxel), "displacement_within_map_error"),
        (pair["change_supported"], "no_repeatable_simsac_change"),
        (len(free_frames(origin)) >= 2, "origin_vacancy_unobserved"),
        (len(free_frames(destination)) >= 2, "destination_vacancy_unobserved")]:
        if not good:
            reasons.append(reason)
    return reasons


def unique_assignments(pairs, minimum=.0):
    if not pairs:
        return []
    left, right = sorted({p["before"] for p in pairs}), sorted({p["after"] for p in pairs})
    scores = np.zeros((len(left), len(right)+len(left)))
    lookup = {}
    for p in pairs:
        if p["score"] > minimum:
            i, j = left.index(p["before"]), right.index(p["after"])
            scores[i, j] = p["score"] - minimum
            lookup[(i, j)] = p
    return [lookup[(i, j)] for i, j in zip(*linear_sum_assignment(-scores)) if (i, j) in lookup]


def surface_sample(vertices, faces, size=512, face_regions=None):
    triangles = vertices[faces]
    areas = np.linalg.norm(np.cross(triangles[:, 1]-triangles[:, 0], triangles[:, 2]-triangles[:, 0]), axis=1)/2
    if areas.sum() <= 1e-10:
        points = triangles.reshape(-1, 3)
        return (points, np.repeat(face_regions, 3)) if face_regions is not None else points
    rng = np.random.default_rng(0)
    indices = rng.choice(len(triangles), size=size, p=areas/areas.sum())
    selected = triangles[indices]
    u, v = np.sqrt(rng.random(size)), rng.random(size)
    points = (1-u[:, None])*selected[:, 0] + (u*(1-v))[:, None]*selected[:, 1] + (u*v)[:, None]*selected[:, 2]
    return (points, face_regions[indices]) if face_regions is not None else points


def opposite_observations(points, epoch, views, tolerance, point_regions=None, label=None):
    records = defaultdict(list)
    for view in views:
        if view["epoch"] == epoch:
            continue
        shape = view["depth"].shape
        ij, z, inside = project_in_view(points, view, point_regions)
        indices = np.flatnonzero(inside)
        if not len(indices):
            continue
        # Distinct projected pixels, rather than repeated samples on one ray.
        _, unique = np.unique(ij[indices], axis=0, return_index=True)
        indices = indices[unique]
        x, y = ij[indices].T
        valid = view["valid"][y, x] & ~view["person"][y, x]
        if 'point_region' in view:
            valid &= view['point_region'][y, x] == point_regions[indices]
        gap = view["depth"][y, x] - z[indices]
        free, occluded = valid & (gap > tolerance), valid & (gap < -tolerance)
        occupied = valid & ~free & ~occluded
        labels=view.get('segmentation_labels', [])
        same_class = valid & (view['segmentation_class'][y,x] == labels.index(label)+1) if 'segmentation_class' in view and label in labels else np.zeros(len(indices),bool)
        n = len(indices)
        lo, hi = ij[indices].min(0), ij[indices].max(0)+1
        records[view["frame_id"]].append(dict(frame_id=view["frame_id"], view=view["id"],
              pixels=n, observed_pixels=int(valid.sum()), free_fraction=float(free.sum()/n),
              occluded_fraction=float(occluded.sum()/n), occupied_fraction=float(occupied.sum()/n),
              unknown_fraction=float((~valid).sum()/n),
              same_class_fraction=float(same_class.sum()/n),
              segmentation_available='segmentation_class' in view and label in labels,
              usable=bool(valid.sum() >= POLICY["min_projected_pixels"] and inside.mean() >= POLICY["min_projected_fraction"]),
              bbox=np.r_[lo, hi].tolist(), role="expected_location"))
    # Choose the face by observed coverage, never by its free-space verdict.
    return [max(group, key=lambda r: (r["observed_pixels"], r["pixels"], r["view"])) for _, group in sorted(records.items())]


def classify(root, manifest_path):
    root, manifest_path = Path(root), Path(manifest_path).resolve()
    prepared, original = read(root / "manifest.json"), read(manifest_path)
    changes = read(root / "results/changes.json")
    appearance_meta = read(root / "results/object-appearance.json")
    if appearance_meta["source_manifest_sha256"] != sha(root / "manifest.json"):
        raise ValueError("Appearance descriptors refer to different input geometry")
    if appearance_meta["descriptors_sha256"] != sha(root / "results/object-appearance.npz"):
        raise ValueError("Appearance descriptors changed")
    descriptors = dict(np.load(root / "results/object-appearance.npz"))
    objects, views, meshes = {}, [], []
    for meta in prepared["views"]:
        path = root / "views" / (meta["id"] + ".npz")
        if sha(path) != meta["geometry_sha256"]:
            raise ValueError("Prepared geometry changed")
        view = dict(np.load(path))
        view["person"] = np.load(root / "results/views" / (meta["id"] + ".npz"))["person"]
        views.append({**meta, **view})
    for epoch, spec in enumerate([original["before"], original["after"]]):
        path = Path(spec["mesh"])
        if not path.is_absolute():
            path = manifest_path.parent / path
        vertices, mesh = aligned_mesh(root, prepared, spec, epoch, manifest_path.parent)
        meshes.append((vertices, mesh["faces"], mesh["face_instance_id"]))
        for record in [o for o in changes["objects"] if o["epoch"] == epoch]:
            key = f'E{epoch}-I{record["instance_id"]}'
            selected = np.flatnonzero(mesh["face_instance_id"] == record["instance_id"])
            if 'face_region' in mesh:
                points, point_regions = surface_sample(vertices, mesh['faces'][selected], face_regions=mesh['face_region'][selected])
            else:
                points, point_regions = surface_sample(vertices, mesh['faces'][selected]), None
            observations = appearance_meta["objects"].get(key, {}).get("observations", [])
            objects[key] = dict(**record, id=key, points=points, point_regions=point_regions, original_faces=selected,
                                center=np.median(points, 0).tolist(), own_observations=observations,
                                segmented_frames=len({o['frame_id'] for o in observations if o.get('segmentation_coverage') is not None and o['segmentation_coverage'] >= .35}),
                                identity_frames=len(descriptors.get(key, [])), descriptor=descriptors.get(key))
            local_error, supported = object_uncertainty(prepared['registration'], record.get('regions'), prepared['settings']['voxel_size_u'])
            objects[key].update(map_error_u=local_error, alignment_supported=supported)
    voxel = prepared["settings"]["voxel_size_u"]
    error, _ = object_uncertainty(prepared['registration'], [], voxel)
    tolerance = max(voxel, 2*error)
    vacancy_tolerance = max(prepared["settings"]["visibility_tolerance_u"], 3*error)
    opposite = {}
    active = [k for k, o in objects.items() if o["state"] == "change_candidate"]
    for key in active:
        opposite[key] = opposite_observations(objects[key]["points"], objects[key]["epoch"], views,
                                              max(prepared['settings']['visibility_tolerance_u'], objects[key]['map_error_u']), objects[key]['point_regions'], objects[key]['label'])
    before, after = [[o for o in objects.values() if o["epoch"] == epoch] for epoch in [0, 1]]
    stationary_options = []
    for a in before:
        for b in after:
            if a["label"] != b["label"] and "unknown" not in [a["label"], b["label"]]:
                continue
            pair_tolerance=max(voxel, a['map_error_u'], b['map_error_u'])
            f1, directional = coverage(a["points"], b["points"], pair_tolerance)
            appearance = identity_score(a["descriptor"], b["descriptor"])
            if f1 >= POLICY["min_stationary_f1"] and min(directional) >= .35 and appearance >= .70:
                stationary_options.append(dict(before=a["id"], after=b["id"], score=.65*f1+.35*appearance,
                                                shape_f1=f1, appearance=appearance, directional_coverage=directional,
                                                shape_tolerance_u=pair_tolerance,
                                                frame_support=min(a['identity_frames'],b['identity_frames'])))
    for pair in stationary_options:
        others=[p['score'] for p in stationary_options if p is not pair and (p['before']==pair['before'] or p['after']==pair['after'])]
        pair['margin']=pair['score']-max(others,default=0)
    stationary = unique_assignments(stationary_options)
    protected = {key for p in stationary for key in [p["before"], p["after"]]}
    comparisons = []
    for a in before:
        if a["id"] in protected:
            continue
        for b in after:
            if b["id"] in protected or (a["label"] != b["label"] and "unknown" not in [a["label"], b["label"]]) or (a["id"] not in active and b["id"] not in active):
                continue
            # Rigid fitting removes alignment displacement; its shape gate
            # must not grow with unrelated scene registration residuals.
            fit = rigid_fit(a["points"], b["points"], max(voxel, error))
            appearance = identity_score(a["descriptor"], b["descriptor"])
            m = np.array(fit["before_to_after_rigid"])
            displacement = transform(np.array(a["center"])[None], m)[0]-a["center"]
            comparisons.append(dict(**fit, before=a["id"], after=b["id"], appearance=appearance,
                  map_error_u=max(a['map_error_u'], b['map_error_u']),
                  alignment_supported=a['alignment_supported'] and b['alignment_supported'],
                  displacement_vector_u=displacement.tolist(), displacement_u=float(np.linalg.norm(displacement)),
                  frame_support=min(a["identity_frames"], b["identity_frames"]),
                  change_supported=a["id"] in active or b["id"] in active,
                  score=.6*fit["shape_f1"]+.4*appearance))
    for pair in comparisons:
        competitors = [p["score"] for p in comparisons if p is not pair and (p["before"] == pair["before"] or p["after"] == pair["after"])]
        pair["margin"] = float(pair["score"]-max(competitors, default=0))
        for key in [pair["before"], pair["after"]]:
            if key not in opposite:
                opposite[key] = opposite_observations(objects[key]["points"], objects[key]["epoch"], views,
                                                      max(prepared['settings']['visibility_tolerance_u'], objects[key]['map_error_u']), objects[key]['point_regions'], objects[key]['label'])
        pair["rejection_reasons"] = movement_rejections(pair, opposite[pair["before"]], opposite[pair["after"]], pair['map_error_u'], voxel)
    moves = unique_assignments([p for p in comparisons if not p["rejection_reasons"]])
    events, consumed = [], set()

    def add_event(kind, keys, reason, matching=None):
        own = [objects[k] for k in keys]
        bounds = np.stack([np.min([o["bounds"][0] for o in own], 0), np.max([o["bounds"][1] for o in own], 0)])
        photos = {}
        for epoch in [0, 1]:
            present = next((o for o in own if o["epoch"] == epoch), None)
            if present is not None and present["own_observations"]:
                observations = sorted(present["own_observations"], key=lambda o: (-o["pixels"], o["view"]))
                photos[str(epoch)] = dict(**observations[0], role="observed_object", object_id=present["id"])
            elif present is None:
                candidates = opposite.get(own[0]["id"], [])
                free = free_frames(candidates)
                candidates = sorted(free or candidates, key=lambda o: (-o["observed_pixels"], o["view"]))
                if candidates:
                    photos[str(epoch)] = candidates[0]
        event = dict(id=f'EV-{len(events)+1:03d}', type=kind, label=own[0]["label"], object_ids=keys,
                     registration_error_u=max(o['map_error_u'] for o in own),
                     alignment_supported=all(o['alignment_supported'] for o in own),
                     before=next((o["id"] for o in own if o["epoch"] == 0), None),
                     after=next((o["id"] for o in own if o["epoch"] == 1), None),
                     bounds=bounds.tolist(), reason_code=reason, photos=photos, matching=matching,
                     vacancy={o["id"]: opposite.get(o["id"], []) for o in own},
                     segmented_background_frames={o['id']: segmented_background_frames(opposite.get(o['id'], [])) for o in own},
                     source_change_frames={o["id"]: o["changed_frames"] for o in own},
                     automatic=True, manual_overrides_used=False, physical_identity_verified=False)
        events.append(event)
        consumed.update(keys)

    for pair in moves:
        add_event("moved", [pair["before"], pair["after"]], "rigid_identity_and_both_vacancies_supported", pair)
    for pair in stationary:
        keys = [pair["before"], pair["after"]]
        if any(k in active for k in keys):
            if pair['frame_support'] < POLICY['minimum_identity_frame_support']:
                add_event('unresolved', keys, 'segmented_identity_observation_insufficient', pair)
            elif pair['margin'] < POLICY['min_match_margin']:
                add_event('unresolved', keys, 'counterpart_identity_ambiguous', pair)
            elif not all(k in active for k in keys):
                add_event('unresolved', keys, 'one_sided_rgb_difference', pair)
            elif all(objects[k]['alignment_supported'] for k in keys):
                add_event("modified", keys, "same_location_object_with_repeatable_simsac_surface_change", pair)
            else:
                add_event('unresolved', keys, 'registration_uncertainty_unsupported', pair)
    for key in active:
        if key in consumed:
            continue
        obj = objects[key]
        plausible = [p for p in comparisons if key in [p["before"], p["after"]] and p["shape_f1"] >= .55 and p["appearance"] >= .70]
        kind, reason = decide_unmatched(obj["epoch"], True, obj["identity_frames"], opposite[key], bool(plausible), obj['segmented_frames'])
        if not obj['alignment_supported']:
            kind, reason = 'unresolved', 'registration_uncertainty_unsupported'
        add_event(kind, [key], reason, sorted(plausible, key=lambda p: -p["score"])[:3] or None)
    registry = {}
    geometry = root / "events/objects"
    geometry.mkdir(parents=True, exist_ok=True)
    for key in sorted(consumed):
        o = objects[key]
        vertices, faces, ids = meshes[o["epoch"]]
        p = vertices[faces[o["original_faces"]]].reshape(-1, 3).astype("<f4")
        (geometry / (key + ".bin")).write_bytes(p.tobytes())
        registry[key] = {k: v for k, v in o.items() if k not in ["points", "point_regions", "descriptor", "original_faces", "own_observations"]}
        registry[key].update(mesh=f'events/objects/{key}.bin', vertices=len(p), original_face_ids=o["original_faces"].tolist())
    counts = {kind: sum(e["type"] == kind for e in events) for kind in ["added", "moved", "removed", "modified", "unresolved"]}
    result = dict(schema="mesh-simsac-events-v1", events=events, objects=registry, counts=counts,
                  policy={**POLICY, "shape_tolerance_u": tolerance, "vacancy_tolerance_u": vacancy_tolerance,
                          "registration_error_u": error, "minimum_translation_u": max(3*error, 3*voxel)},
                  stationary_pairs=stationary, movement_comparisons=comparisons,
                  source_changes_sha256=sha(root / "results/changes.json"),
                  alignment_summary=dict(new_alignment_fitted=prepared['registration']['new_alignment_fitted'],
                       independent_pointcloud=prepared['registration'].get('independent_pointcloud'),
                       pointcloud_overlap=prepared['registration']['overlap_fractions'],
                       pointcloud_residual_p90_u=prepared['registration'].get('pointcloud_residual_p90_u'),
                       region_count=prepared['registration'].get('region_count', 1),
                       held_out_validation=bool(prepared['registration'].get('fresh_structural'))),
                  appearance_sha256=sha(root / "results/object-appearance.npz"),
                  gaussian_inputs_used=False, keypoint_plan_used=False, existing_event_decisions_used=False,
                  scope=prepared["scope"], accuracy_claimed=False)
    write(root / "results/events.json", result)
    print(counts, flush=True)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    classify(args.root, args.manifest)
