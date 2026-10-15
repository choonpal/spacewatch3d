"""Link real photo candidates to surface regions, preserving camera uncertainty."""
import argparse
import json
from pathlib import Path
import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree
from build_rescan_review import read_cloud, TO_DISPLAY, surface_distance, digest
from recover_rescan_keyframes import render


def project(points, c2w, focal):
    camera = (points - c2w[:3, 3]) @ c2w[:3, :3]
    z = camera[:, 2]
    pixels = camera[:, :2] / np.maximum(z[:, None], 1e-9) * focal + [323.5, 241.5]
    return pixels, z


def link(data, output):
    path = output / "review.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    clouds = [read_cloud(data / f"scene_i_{e}.ply") for e in (1, 2)]
    transform = np.array(manifest["registration"]["after_to_before"])
    clouds[1]["xyz"] = clouds[1]["xyz"] @ transform[:3,:3].T + transform[:3,3]
    meshes = [o3d.io.read_triangle_mesh(str(output / f"cache/{epoch}-depth8.ply")) for epoch in (0, 1)]
    frames = manifest["keyframes"]
    samples = {}
    for region in manifest["regions"]:
        ids = np.fromfile(output / f"region-{region['id']:03d}.indices.bin", dtype="<u4")
        ids = ids[np.linspace(0, len(ids)-1, min(1500,len(ids))).astype(int)]
        samples[region["id"]] = clouds[region["epoch"]]["xyz"][ids]
        region["keyframes"] = [[], []]
        # Sensitivity to small residual alignment changes, not sensor-noise GT.
        fractions = []
        xyz = samples[region["id"]]
        for shift in ([.005,0,0],[-.005,0,0],[0,0,.005],[0,0,-.005]):
            d = surface_distance(meshes[1-region["epoch"]], xyz + shift)
            fractions.append(float(np.mean(d > manifest["threshold_m"])))
        region["alignment_survival_min"] = min(fractions)
        region["alignment_sensitive"] = min(fractions) < .8
        region["sample_scope"] = "up to 1500 original candidate points; ±5mm translations"
    for f in frames:
        checked = f["status"] == "estimated_rgb_checked"
        c2w = np.array(f["camera_to_world"] if checked else f.get("depth_camera_to_world", []))
        if c2w.shape != (4,4):
            continue
        focal = f["focal_px"] if checked else 585.225
        cached = output / "frames-cache" / (f["id"] + "-visibility.npy")
        if cached.exists():
            depth = np.load(cached)
        else:
            _, _, depth = render(meshes[f["epoch"]], c2w, focal)
            np.save(cached, depth)
        for region in manifest["regions"]:
            pixels, z = project(samples[region["id"]], c2w, focal)
            inside = (z > .25) & (pixels[:,0] >= 20) & (pixels[:,0] < 628) & (pixels[:,1] >= 20) & (pixels[:,1] < 464)
            if np.count_nonzero(inside) < 8:
                continue
            coords = np.rint(pixels[inside]).astype(int)
            surface_z = depth[coords[:,1],coords[:,0]]
            # Finite opposite observed mesh required. Infinity is unknown,
            # never an observation of empty space.
            visible = np.isfinite(surface_z) & (surface_z >= z[inside] - .10)
            fraction = float(np.mean(visible) * np.mean(inside))
            if visible.sum() < 8 or fraction < .025:
                continue
            relevant = pixels[inside][visible]
            box = np.r_[relevant.min(0), relevant.max(0)].tolist()
            area = np.prod(np.maximum(relevant.max(0)-relevant.min(0), 1)) / (648*484)
            score = fraction * np.sqrt(area) * (1.7 if checked else 1)
            region["keyframes"][f["epoch"]].append({"frame_id": f["id"], "score": float(score),
                "visible_sample_fraction": fraction, "bbox_px": box if checked else None,
                "camera_check": "rgb_feature_correspondence" if checked else "depth_registration_only",
                "relation": "same_surface_view_candidate" if f["epoch"] == region["epoch"] else "same_location_review_candidate",
                "absence_confirmed": False})
    for region in manifest["regions"]:
        for epoch in (0,1):
            region["keyframes"][epoch].sort(key=lambda x: (-int(x["camera_check"] == "rgb_feature_correspondence"), -x["score"]))
            region["keyframes"][epoch] = region["keyframes"][epoch][:5]
        region["paired_photos"] = all(region["keyframes"])
        region["paired_rgb_checked"] = all(any(f["camera_check"] == "rgb_feature_correspondence" for f in group) for group in region["keyframes"])
    # Area is measured on the supported full mesh, with labels transferred from
    # raw geometric regions solely for reporting. Triangulation is not compared.
    for epoch, mesh in enumerate(meshes):
        ids = np.zeros(len(clouds[epoch]["xyz"]), np.uint8)
        for region in manifest["regions"]:
            if region["epoch"] == epoch:
                index = np.fromfile(output / f"region-{region['id']:03d}.indices.bin", dtype="<u4")
                ids[index] = region["id"]
        vertices, faces = np.asarray(mesh.vertices), np.asarray(mesh.triangles)
        nearest = cKDTree(clouds[epoch]["xyz"]).query(vertices)[1]
        face_ids = ids[nearest[faces]]
        area = np.linalg.norm(np.cross(vertices[faces[:,1]]-vertices[faces[:,0]], vertices[faces[:,2]]-vertices[faces[:,0]]),axis=1)/2
        for region in manifest["regions"]:
            if region["epoch"] == epoch:
                region["supported_area_m2"] = float(np.sum(area * np.mean(face_ids == region["id"],axis=1)))
        tri = faces[0]
        manifest["displays"][epoch]["mesh_example"] = {"face_indices": tri.tolist(), "vertices_m": vertices[tri].tolist(),
                                                       "has_vertex_color": mesh.has_vertex_colors(), "has_normals": mesh.has_vertex_normals()}
    manifest["link_summary"] = {"sampled_original_frames": len(frames),
        "rgb_feature_checked_frames": sum(f["status"] == "estimated_rgb_checked" for f in frames),
        "depth_registered_frames": sum(f["status"] != "unlocalized" for f in frames),
        "regions_with_before_after_photo_candidates": sum(r["paired_photos"] for r in manifest["regions"]),
        "regions_with_both_rgb_checked": sum(r["paired_rgb_checked"] for r in manifest["regions"]),
        "camera_ground_truth_available": False}
    manifest["label_independence"] = "Candidate masks consume XYZ/surface distances only; instance/class labels supply context after discovery"
    path.write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding="utf-8")
    print(manifest["link_summary"], flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    args = p.parse_args()
    link(args.data,args.output)
