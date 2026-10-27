"""Check input provenance, raw probability accounting and independent-frame votes."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from .core import project_in_view, read, sha, write


def validate(root, check_sources=True):
    root = Path(root)
    manifest = read(root / "manifest.json")
    changes = read(root / "results/changes.json")
    model = read(root / "results/model.json")
    assert read(root / "inference-status.json")["state"] == "complete"
    assert model["loaded_strict"] and model["model"] == "SimSaC"
    assert changes["source_manifest_sha256"] == sha(root / "manifest.json")
    for data in [manifest, changes]:
        assert not any(data[k] for k in ["gaussian_inputs_used", "keypoint_plan_used", "existing_change_results_used"])
    if check_sources:
        for path, expected in manifest["source_hashes"].items():
            assert sha(path) == expected, path
    by_id = {v["id"]: v for v in manifest["views"]}
    prepared = {}
    regional_views = 0
    for vid, meta in by_id.items():
        path = root / "views" / (vid + ".npz")
        assert sha(path) == meta["geometry_sha256"]
        with np.load(path) as data:
            assert not (data["valid"] & ~data["mesh_hit"]).any()
            assert not (data["valid"] & ~data["pointcloud_supported"]).any()
            assert not (data["valid"] & data["pointcloud_occluded"]).any()
            prepared[vid] = data["rgb"].shape[:2]
            if manifest['registration'].get('regional_alignment_applied'):
                assert 'point_region' in data and 'region_poses' in data
                mask = data['valid']
                y, x = np.nonzero(mask)
                ij, z, inside = project_in_view(data['points'][mask], dict(data), data['point_region'][mask])
                # float32 ray coordinates can land a few ulps beyond an image
                # edge. Exact pixel/depth roundtrip still applies to every ray.
                interior = (x > 0) & (y > 0) & (x < data['depth'].shape[1]-1) & (y < data['depth'].shape[0]-1)
                assert inside[interior].all(), 'Regional projection left its original image'
                np.testing.assert_array_equal(ij, np.column_stack([x, y]))
                np.testing.assert_allclose(z, data['depth'][mask], atol=2e-4, rtol=2e-5)
                regional_views += 1
    for cache in manifest['registration'].get('aligned_meshes', []):
        assert sha(root / cache['file']) == cache['sha256']
    count, pixels = 0, 0
    for audit in read(root / "results/pair-audit.json"):
        path = root / audit["path"]
        assert sha(path) == audit["sha256"]
        assert by_id[audit["query"]]["epoch"] != by_id[audit["reference"]]["epoch"]
        assert by_id[audit["query"]]["rgb_sha256"] == audit["query_rgb_sha256"]
        assert by_id[audit["reference"]]["rgb_sha256"] == audit["reference_rgb_sha256"]
        with np.load(path) as data:
            p = data["probability"]
            assert p.shape == prepared[audit["query"]] and np.isfinite(p).all()
            assert ((p >= 0) & (p <= 1)).all()
            np.testing.assert_array_equal(data["raw"], p > manifest["settings"]["probability_threshold"])
            parts = [data[k] for k in ["candidate", "excluded_person", "unsupported_geometry", "occluded", "unobserved_counterpart"]]
            np.testing.assert_array_equal(sum(m.astype(int) for m in parts), data["raw"].astype(int))
            assert not (data["candidate"] & ~data["observable"]).any()
            for key, value in audit["pixels"].items():
                assert int(data[key].sum()) == value
            pixels += int(data["candidate"].sum())
        count += 1
    settings = manifest["settings"]
    for pair in read(root / "pairs.json"):
        frames = [by_id[r["id"]]["frame_id"] for r in pair["refs"]]
        assert len(set(frames)) == len(frames), "Cube views of one reference frame cannot count twice"
    for obj in changes["objects"]:
        obs, pos = set(obj["observed_frames"]), set(obj["changed_frames"])
        assert pos <= obs
        if obj["state"] == "change_candidate":
            assert len(pos) >= settings["min_frames"]
            assert len(pos) / len(obs) >= settings["changed_frame_ratio"]
        if not obs:
            assert obj["state"] == "insufficient_observations"
    for region in changes["regions"]:
        assert len(set(region["source_frames"])) >= settings["min_frames"]
        assert not region["verified_object_change"]
    for epoch in [0, 1]:
        audit = read(root / f"pointcloud-{epoch}-audit.json")
        if audit["source_is_mesh_sample"]:
            assert audit["support_mode"] == "mesh_first_hit" and audit["support_points"] == 0
    result = dict(state="passed", prepared_views=len(by_id), fresh_inference_pairs=count,
                  supported_pair_candidate_pixels=pixels, regions=len(changes["regions"]),
                  source_files_checked=check_sources, gaussian_inputs_used=False,
                  regional_camera_roundtrip_views=regional_views,
                  original_probabilities_account_exactly=True, independent_frame_votes=True,
                  ground_truth_accuracy_claimed=False)
    write(root / "validation.json", result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--skip-source-files", action="store_true", help="For a remote snapshot whose original sources live on another host")
    args = parser.parse_args()
    print(validate(args.root, not args.skip_source_files))
