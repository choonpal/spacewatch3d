"""Existing automatic identity/free-space event gates; independent of capture IDs."""
import numpy as np
from scipy.spatial import cKDTree
from scipy.optimize import linear_sum_assignment
from .core import transform

POLICY = dict(min_frames=2, min_shape_f1=.75, min_shape_direction=.60,
              min_appearance=.80, min_match_margin=.06, min_stationary_f1=.55,
              min_free_fraction=.65, max_occluded_fraction=.20,
              min_projected_pixels=12, min_projected_fraction=.20,
              minimum_identity_frame_support=2, object_scale_fitted=False,
              yaw_seeds_degrees=list(range(0, 360, 45)), manual_overrides_used=False)

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

def decide_unmatched(epoch, has_change, own_frames, opposite, plausible_match):
    if not has_change:
        return "unchanged", "no_repeatable_simsac_change"
    if own_frames < POLICY["min_frames"]:
        return "unresolved", "own_observation_insufficient"
    if plausible_match:
        return "unresolved", "counterpart_identity_ambiguous"
    if sum(r["usable"] and r["occupied_fraction"] >= .45 for r in opposite) >= 2:
        return "unresolved", "surface_still_occupies_expected_location"
    if len(free_frames(opposite)) >= POLICY["min_frames"]:
        return ("removed" if epoch == 0 else "added"), "expected_location_observed_free_in_independent_frames"
    return "unresolved", "opposite_location_occluded_or_unobserved"

def movement_rejections(pair, origin, destination, error, voxel):
    reasons = []
    for good, reason in [
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
