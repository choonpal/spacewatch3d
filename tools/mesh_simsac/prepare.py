"""Prepare original video/image, segmented mesh and point-cloud inputs automatically."""
from __future__ import annotations

import argparse
import hashlib
import time
from pathlib import Path

import numpy as np
import open3d as o3d
from PIL import Image
from scipy.ndimage import minimum_filter
from scipy.spatial import cKDTree

from .core import (camera_transform, choose_pairs, cube_rotations, erp_project, intrinsics,
                  project, rays, read, sha, similarity, transform, write)
from .regional import load_regional, FreshRegionalAlignment


def resolve(base, value):
    path = Path(value)
    return path if path.is_absolute() else (base / path).resolve()


def load_cloud(path):
    if path.suffix.lower() == ".npz":
        with np.load(path, allow_pickle=False) as data:
            points = data["points"].astype(float)
    else:
        points = np.asarray(o3d.io.read_point_cloud(str(path)).points)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("Point cloud must have shape N x 3")
    points = points[np.isfinite(points).all(1)]
    if len(points) < 30:
        raise ValueError("Insufficient finite point-cloud points")
    return points


def downsample(points, voxel):
    cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
    return cloud.voxel_down_sample(voxel)


def register_clouds(sides, settings, regional=None):
    voxel = settings["voxel_size_u"]
    mode = settings.get("registration", "auto")
    if isinstance(regional, FreshRegionalAlignment):
        matrices = [np.eye(4), regional.base]
        details = dict(mode='auto_structural', new_alignment_fitted=True,
                       fresh_structural=regional.report)
    elif mode == "provided":
        matrices = [similarity(s["spec"]["common_from_world"])[0] for s in sides]
        details = dict(mode=mode, new_alignment_fitted=False)
    elif mode == "auto":
        # Equal map units are required. Metric scale is never invented from RGB.
        o3d.utility.random.seed(0)
        clouds, features = [], []
        reg = o3d.pipelines.registration
        for side in sides:
            cloud = downsample(side["points_world"], voxel * 3)
            if len(cloud.points) < 30:
                raise ValueError("Not enough points for automatic registration")
            cloud.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=voxel * 6, max_nn=30))
            features.append(reg.compute_fpfh_feature(cloud, o3d.geometry.KDTreeSearchParamHybrid(radius=voxel * 15, max_nn=100)))
            clouds.append(cloud)
        initial = reg.registration_ransac_based_on_feature_matching(
            clouds[1], clouds[0], features[1], features[0], True, voxel * 5,
            reg.TransformationEstimationPointToPoint(False), 3,
            [reg.CorrespondenceCheckerBasedOnEdgeLength(.9), reg.CorrespondenceCheckerBasedOnDistance(voxel * 5)],
            reg.RANSACConvergenceCriteria(50000, .999))
        refined = reg.registration_icp(clouds[1], clouds[0], voxel * 3, initial.transformation,
                                       reg.TransformationEstimationPointToPlane(), reg.ICPConvergenceCriteria(max_iteration=60))
        matrices = [np.eye(4), similarity(refined.transformation)[0]]
        details = dict(mode=mode, new_alignment_fitted=True, ransac_fitness=float(initial.fitness),
                       icp_fitness=float(refined.fitness), icp_rmse_u=float(refined.inlier_rmse),
                       physically_correct_alignment_verified=False)
    else:
        raise ValueError("registration must be 'provided' or 'auto'")
    aligned = []
    for epoch, (side, matrix) in enumerate(zip(sides, matrices)):
        points = transform(side['points_world'], matrix)
        if regional:
            points = regional.move(points, regional.assign(side['points_world'], epoch), epoch)
        aligned.append(np.asarray(downsample(points, voxel).points))
    if regional:
        details.update(regional_alignment_applied=True, region_count=len(regional.corrections)-1,
                       region_matrices=[regional.matrices(e).tolist() for e in [0, 1]],
                       projection='Original first-hit rays; per-region camera transforms for cross-capture projection',
                       object_specific_alignment=False, residual_translation_bias_applied=isinstance(regional, FreshRegionalAlignment))
    fractions, errors, p90 = [], [], []
    for a, b in [(aligned[0], aligned[1]), (aligned[1], aligned[0])]:
        distances = cKDTree(b).query(a, workers=4)[0]
        good = distances <= settings["visibility_tolerance_u"]
        fractions.append(float(good.mean()))
        errors.append(float(np.median(distances[good])) if good.any() else None)
        # Uncensored whole-cloud quantile: large unexplained areas must not
        # disappear from the error estimate merely because they miss the gate.
        p90.append(float(np.quantile(distances, .9)))
    details.update(common_from_world=[m.tolist() for m in matrices], overlap_fractions=fractions,
                   median_inlier_distances_u=errors, pointcloud_residual_p90_u=p90,
                   independent_pointcloud=["pointcloud" in s['spec'] for s in sides],
                   tolerance_u=settings["visibility_tolerance_u"],
                   state="passed_geometric_gate" if min(fractions) >= settings.get("min_registration_overlap", .2) else "insufficient_overlap",
                   caveat="Overlap gate does not resolve repeated-room/floor ambiguity or prove registration correctness")
    return matrices, details


def frame_image(spec, frame, base, video_capture):
    if "image" in frame:
        return np.asarray(Image.open(resolve(base, frame["image"])).convert("RGB"))
    if "video" not in spec or "source_frame_index" not in frame:
        raise ValueError("Each frame needs an image path or video + source_frame_index")
    import cv2
    if video_capture[0] is None:
        video_capture[0] = cv2.VideoCapture(str(resolve(base, spec["video"])))
    video_capture[0].set(cv2.CAP_PROP_POS_FRAMES, int(frame["source_frame_index"]))
    ok, image = video_capture[0].read()
    if not ok:
        raise ValueError(f"Cannot decode video frame {frame['source_frame_index']}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def point_depth(points, pose, K, shape):
    ij, z, inside = project(points, pose, K, shape)
    depth = np.full(shape, np.inf)
    np.minimum.at(depth, (ij[inside, 1], ij[inside, 0]), z[inside])
    # One-pixel splat catches a point-cloud foreground that the coarser mesh missed.
    return minimum_filter(depth, size=3, mode="constant", cval=np.inf)


def rasterize(vertices, faces, ids, cloud, pose, K, shape, tolerance, scene=None, tree=None):
    if scene is None:
        scene = o3d.t.geometry.RaycastingScene(nthreads=4)
        scene.add_triangles(o3d.core.Tensor(vertices.astype("f4")), o3d.core.Tensor(faces.astype("u4")))
    directions = rays(K, shape)
    world_directions = directions @ pose[:3, :3].T
    cast = np.concatenate([np.broadcast_to(pose[:3, 3], world_directions.shape), world_directions], -1)
    hit = scene.cast_rays(o3d.core.Tensor(cast.astype("f4")), nthreads=4)
    depth = hit["t_hit"].numpy()
    triangle = hit["primitive_ids"].numpy()
    hit_valid = np.isfinite(depth) & (depth > 0) & (triangle < len(faces))
    depth = np.where(hit_valid, depth, 0).astype("f4")
    points = pose[:3, 3] + world_directions * depth[..., None]
    support = hit_valid.copy()
    foreground = np.zeros(shape, bool)
    if cloud is not None:
        tree = tree or cKDTree(cloud)
        support[:] = False
        if hit_valid.any():
            support[hit_valid] = tree.query(points[hit_valid], workers=4)[0] <= tolerance
        pc_depth = point_depth(cloud, pose, K, shape)
        foreground = np.isfinite(pc_depth) & (depth > pc_depth + tolerance)
    valid = hit_valid & support & ~foreground
    instance = np.full(shape, -1, "i4")
    face = np.full(shape, -1, "i4")
    instance[hit_valid] = ids[triangle[hit_valid]]
    face[hit_valid] = triangle[hit_valid].astype("i4")
    return dict(depth=depth, points=points.astype("f4"), instance=instance, face=face,
                valid=valid, mesh_hit=hit_valid, pointcloud_supported=support,
                pointcloud_occluded=foreground, pose=pose, K=K)


def prepare(manifest_path, output, capturer_checkpoint=None, device='cuda'):
    manifest_path, output = Path(manifest_path).resolve(), Path(output).resolve()
    manifest, base = read(manifest_path), manifest_path.parent
    if manifest.get("schema") != "mesh-simsac-input-v1":
        raise ValueError("Expected mesh-simsac-input-v1 manifest")
    settings = dict(size=224, frame_stride=3, voxel_size_u=.05, visibility_tolerance_u=.15,
                    probability_threshold=.5, min_overlap=.08, max_refs=2, min_frames=2,
                    changed_frame_ratio=.5, object_pixel_ratio=.02, min_object_pixels=16,
                    registration="auto", min_registration_overlap=.2, regional_correction=True,
                    registration_samples=60000, region_width_heights=2.5, max_regions=64)
    settings.update(manifest.get("settings", {}))
    from .observation_quality import quality_settings
    quality_settings(settings)
    capturer_detector, capturer_records = None, []
    for key in ["size", "frame_stride", "max_refs", "min_frames", "min_object_pixels"]:
        if not isinstance(settings[key], int) or settings[key] < 1:
            raise ValueError(f"{key} must be a positive integer")
    for key in ["voxel_size_u", "visibility_tolerance_u"]:
        if not np.isfinite(settings[key]) or settings[key] <= 0:
            raise ValueError(f"{key} must be positive and finite")
    for key in ["probability_threshold", "min_overlap", "changed_frame_ratio", "object_pixel_ratio", "min_registration_overlap"]:
        if not 0 < settings[key] <= 1:
            raise ValueError(f"{key} must be in (0, 1]")
    if output.exists():
        raise FileExistsError(f"Use a new output directory: {output}")
    output.mkdir(parents=True)
    write(output / "status.json", dict(stage="prepare", state="running"))
    started, sides, hashes = time.time(), [], {str(manifest_path): sha(manifest_path)}
    try:
        for epoch, spec in enumerate([manifest["before"], manifest["after"]]):
            for field in ["mesh", "pointcloud", "objects", "video"]:
                if field in spec:
                    path = resolve(base, spec[field])
                    hashes[str(path)] = sha(path)
            expected_cloud = spec.get("pointcloud_provenance", {}).get("sha256")
            if expected_cloud and hashes[str(resolve(base, spec["pointcloud"]))] != expected_cloud:
                raise ValueError("Point-cloud export differs from its original-source receipt")
            mesh = dict(np.load(resolve(base, spec["mesh"]), allow_pickle=False))
            vertices, faces = mesh["vertices"].astype(float), mesh["faces"].astype("i8")
            ids = mesh["face_instance_id"].astype("i4")
            if vertices.ndim != 2 or vertices.shape[1] != 3 or not np.isfinite(vertices).all():
                raise ValueError("Mesh vertices must be finite N x 3")
            if faces.ndim != 2 or faces.shape[1] != 3 or ids.shape != (len(faces),) or not len(faces):
                raise ValueError("Need triangular faces and one instance ID per original face")
            if faces.min() < 0 or faces.max() >= len(vertices):
                raise ValueError("Mesh triangle index outside vertex array")
            wm = similarity(spec.get("world_from_mesh", np.eye(4)))[0]
            wp = similarity(spec.get("world_from_pointcloud", np.eye(4)))[0]
            if "pointcloud" in spec:
                points = load_cloud(resolve(base, spec["pointcloud"]))
            else:
                # Mesh surface samples are for registration only, never presented
                # as an independent sensor observation or person evidence.
                from .classify import surface_sample
                points = surface_sample(vertices, faces, size=min(100000, max(4096, len(faces))))
                wp = wm
            objects = read(resolve(base, spec["objects"])) if "objects" in spec else []
            if isinstance(objects, dict):
                objects = objects["instance_records"]
            sides.append(dict(epoch=epoch, spec=spec, vertices_world=transform(vertices, wm), faces=faces,
                              ids=ids, points_world=transform(points, wp), objects=objects,
                              source_point_count=len(points)))
        common = [np.asarray(s['spec'].get('common_from_world', np.eye(4))) for s in sides]
        if manifest.get('regional_alignment') and settings['registration'] != 'provided':
            raise ValueError('Frozen regional alignment requires registration=provided')
        regional, regional_hashes = load_regional(manifest, base, common)
        if settings['registration'] == 'auto' and settings['regional_correction']:
            from tools.mesh_simsac.registration import load_mesh, register
            samples = [load_mesh(dict(mesh=str(resolve(base, side['spec']['mesh'])),
                                      world_from_mesh=side['spec'].get('world_from_mesh', np.eye(4))),
                                 settings['registration_samples'], epoch) for epoch, side in enumerate(sides)]
            report = register(samples, dict(height_u=None, min_overlap=settings['min_registration_overlap'],
                                           region_width_heights=settings['region_width_heights'], max_regions=settings['max_regions']))
            regional = FreshRegionalAlignment(report)
        hashes.update(regional_hashes)
        matrices, registration = register_clouds(sides, settings, regional)
        write(output / "registration.json", registration)
        if registration["state"] != "passed_geometric_gate":
            raise ValueError("Registration has insufficient point-cloud overlap; do not infer absence")
        (output / "views").mkdir()
        metadata, arrays, object_rows = [], [], []
        shape = (settings["size"], settings["size"])
        for side, common in zip(sides, matrices):
            spec, epoch = side["spec"], side["epoch"]
            vertices = transform(side["vertices_world"], common)
            aligned = regional.mesh(side['vertices_world'], side['faces'], side['ids'], common, epoch) if regional else None
            if aligned is not None:
                cache = output / f'aligned-mesh-{epoch}.npz'
                np.savez_compressed(cache, **aligned)
                registration.setdefault('aligned_meshes', []).append(dict(file=cache.name, sha256=sha(cache)))
            cloud = np.asarray(downsample(transform(side["points_world"], common), settings["voxel_size_u"] / 2).points)
            scene = o3d.t.geometry.RaycastingScene(nthreads=4)
            scene.add_triangles(o3d.core.Tensor(vertices.astype("f4")), o3d.core.Tensor(side["faces"].astype("u4")))
            cloud = cloud if "pointcloud" in spec else None
            tree, capture = cKDTree(cloud) if cloud is not None else None, [None]
            person_ids = [int(o["instance_id"]) for o in side["objects"] if str(o.get("label", "")).lower() == "person"]
            object_map = {int(o["instance_id"]): o for o in side["objects"]}
            for iid in sorted(set(side["ids"]) - {0, -1}):
                face = np.flatnonzero(side["ids"] == iid)
                obj_points = (aligned['vertices'][aligned['faces'][face]] if aligned is not None else vertices[side['faces'][face]]).reshape(-1, 3)
                object_rows.append(dict(epoch=epoch, instance_id=int(iid), label=object_map.get(int(iid), {}).get("label", "unknown"),
                                        face_count=len(face), bounds=np.stack([obj_points.min(0), obj_points.max(0)]).tolist(),
                                        regions=np.unique(aligned['face_region'][face]).tolist() if aligned is not None else []))
            frame_ids = [str(f["id"]) for f in spec["frames"]]
            if not frame_ids:
                raise ValueError("Each capture needs at least one calibrated source frame")
            if len(set(frame_ids)) != len(frame_ids):
                raise ValueError("Frame IDs must be unique within each capture")
            try:
                for frame in spec["frames"][::settings["frame_stride"]]:
                    original = frame_image(spec, frame, base, capture)
                    if "image" in frame:
                        path = resolve(base, frame["image"])
                        hashes[str(path)] = sha(path)
                        if frame.get("sha256") and hashes[str(path)] != frame["sha256"]:
                            raise ValueError("Original RGB differs from its input receipt")
                    original_mask = None
                    segmentation = None
                    if 'segmentation' in frame:
                        path=resolve(base, frame['segmentation']);hashes[str(path)]=sha(path)
                        segmentation=dict(np.load(path, allow_pickle=False))
                        if frame.get('segmentation_rgb_sha256') and ('image' not in frame or sha(resolve(base,frame['image'])) != frame['segmentation_rgb_sha256']):
                            raise ValueError('Segmentation belongs to a different original RGB frame')
                        if not all(k in segmentation for k in ['class_id','instance_id','person']):
                            raise ValueError('Segmentation requires class_id, instance_id and person arrays')
                        if len({segmentation[k].shape for k in ['class_id','instance_id','person']}) != 1 or segmentation['class_id'].ndim != 2:
                            raise ValueError('Segmentation arrays must share a two-dimensional image shape')
                    if "person_mask" in frame:
                        path = resolve(base, frame["person_mask"])
                        hashes[str(path)] = sha(path)
                        original_mask = np.asarray(Image.open(path).convert("L")) > 0
                        if original_mask.shape != original.shape[:2]:
                            raise ValueError("Person mask must match original image dimensions")
                    mask_policy = spec.get('person_mask_policy','all_people')
                    if capturer_checkpoint and mask_policy != 'capturer_only':
                        if spec.get('projection','erp') != 'erp':
                            raise ValueError('Perspective inputs require explicit photographer-only masks and person_mask_policy=capturer_only')
                        if capturer_detector is None:
                            from .capturer import CapturerDetector
                            capturer_detector = CapturerDetector(capturer_checkpoint,device)
                        original_mask, audit = capturer_detector.predict(original)
                        capturer_records.append(dict(capture=spec['id'],frame_id=str(frame['id']),**audit))
                        mask_policy = 'capturer_only'
                    pose = camera_transform(np.array(frame["world_from_camera"]), common)
                    projection = spec.get("projection", "erp")
                    rotations = cube_rotations() if projection == "erp" else {"perspective": np.eye(3)}
                    if projection not in ["erp", "perspective"]:
                        raise ValueError("projection must be erp or perspective")
                    for face_name, rotation in rotations.items():
                        p = pose.copy()
                        p[:3, :3] = p[:3, :3] @ rotation
                        if projection == "erp":
                            K = intrinsics(*shape[::-1])
                            direction = rays(K, shape)
                            rgb = erp_project(original, direction, rotation)
                            person = erp_project(original_mask.astype("u1"), direction, rotation, True) > 0 if original_mask is not None else np.zeros(shape, bool)
                        else:
                            # Native camera intrinsics are scaled together with the input image.
                            K = np.asarray(frame["K"], float).copy()
                            if K.shape != (3, 3) or not np.isfinite(K).all() or K[0, 0] <= 0 or K[1, 1] <= 0 or not np.allclose(K[2], [0, 0, 1]):
                                raise ValueError("Expected finite pinhole camera K with positive focal lengths")
                            sx, sy = shape[1] / original.shape[1], shape[0] / original.shape[0]
                            K[0] *= sx
                            K[1] *= sy
                            K[0, 2] += .5 * sx - .5
                            K[1, 2] += .5 * sy - .5
                            rgb = np.asarray(Image.fromarray(original).resize(shape[::-1], Image.Resampling.BILINEAR))
                            person = np.asarray(Image.fromarray(original_mask).resize(shape[::-1], Image.Resampling.NEAREST)) if original_mask is not None else np.zeros(shape, bool)
                        geometry = rasterize(vertices, side["faces"], side["ids"], cloud, p, K, shape,
                                             settings["visibility_tolerance_u"], scene, tree)
                        if segmentation is not None:
                            projected={k:erp_project(v.astype('i4'), direction, rotation, True) if projection=='erp'
                                       else np.asarray(Image.fromarray(v.astype('i4')).resize(shape[::-1],Image.Resampling.NEAREST))
                                       for k,v in segmentation.items() if k in ['class_id','instance_id','person']}
                            if spec.get('person_mask_policy') == 'capturer_only' or mask_policy != 'capturer_only':
                                person |= projected['person'] > 0
                            geometry['segmentation_class']=projected['class_id']
                            geometry['segmentation_instance']=projected['instance_id']
                            # Object evidence requires agreement with its actual
                            # source-image mask, not just a mesh ray's instance ID.
                            geometry['segmentation_supported']=(geometry['instance']<=0)|(projected['instance_id']==geometry['instance'])
                        if regional:
                            geometry = regional.view(geometry, aligned['face_region'], epoch)
                        if mask_policy != 'capturer_only':
                            person |= np.isin(geometry["instance"], person_ids)
                        vid = f"E{epoch}-V{len(metadata):05d}"
                        view = dict(id=vid, frame_id=str(frame["id"]), epoch=epoch, rgb=rgb, person=person, **geometry)
                        np.savez_compressed(output / "views" / (vid + ".npz"), **{k: v for k, v in view.items() if isinstance(v, np.ndarray)})
                        Image.fromarray(rgb).save(output / "views" / (vid + ".jpg"), quality=92)
                        record = dict(id=vid, epoch=epoch, capture_id=spec["id"], frame_id=str(frame["id"]),
                                      timestamp_s=frame.get("timestamp_s"), source_frame_index=frame.get("source_frame_index"),
                                      source_image=frame.get("image"), source_video=spec.get("video"), face=face_name,
                                      rgb_sha256=hashlib.sha256(rgb.tobytes()).hexdigest(),
                                      geometry_sha256=sha(output / "views" / (vid + ".npz")),
                                      segmentation_labels=spec.get('segmentation_labels', []),
                                      person_mask_policy=mask_policy,
                                      valid_pixels=int(geometry["valid"].sum()), person_pixels=int(person.sum()))
                        metadata.append(record)
                        arrays.append(view)
                    print("PREPARE", spec["id"], frame["id"], len(metadata), flush=True)
            finally:
                if capture[0] is not None:
                    capture[0].release()
            write(output / f"pointcloud-{epoch}-audit.json", dict(input_points=side["source_point_count"],
                  support_points=len(cloud) if cloud is not None else 0, source=spec.get("pointcloud"), source_is_mesh_sample="pointcloud" not in spec,
                  support_mode="independent_pointcloud" if cloud is not None else "mesh_first_hit",
                  upstream_provenance=spec.get("pointcloud_provenance")))
        pairs = choose_pairs(arrays, settings["visibility_tolerance_u"], settings["min_overlap"], settings["max_refs"])
        write(output / 'registration.json', registration)
        write(output / 'capturer-audit.json',capturer_records)
        write(output / "pairs.json", pairs)
        write(output / "objects.json", object_rows)
        write(output / "manifest.json", dict(schema="mesh-simsac-prepared-v1", settings=settings, views=metadata,
              scope=manifest.get("scope", "Independent captures; change candidates require review"),
              source_hashes=hashes, gaussian_inputs_used=False, keypoint_plan_used=False,
              existing_change_results_used=False, pair_selection="All sampled original frames; six ERP faces; geometric overlap before inference",
              unlabelled_surfaces_included=True, registration=registration))
        write(output / "status.json", dict(stage="prepare", state="complete", views=len(metadata),
              ready_views=sum(bool(p["refs"]) for p in pairs), seconds=time.time() - started))
    except Exception as error:
        write(output / "status.json", dict(stage="prepare", state="error", error=str(error)))
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    prepare(args.manifest, args.output)
