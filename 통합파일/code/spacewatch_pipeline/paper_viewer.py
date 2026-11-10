"""Connect fresh native plans to current detector evidence in one patrol UI.

The native publisher still verifies all plan/render/geometry receipts. This
wrapper never launches a service: engine.commit does so after saving the receipt.
Cube/crop boxes are projected into ERP only for the same physical source frame.
Other actual detector photographs remain available in the in-viewer drawer.
"""
import copy
import hashlib
import os
from pathlib import Path
import shutil

import numpy as np

from .io import digest, read, result, save, verify


def require(condition, message):
    if not condition:
        raise ValueError(message)


def after_photos(event):
    rows = [pair.get("1") for pair in event.get("photo_sets", [])]
    if isinstance(event.get("photos"), dict):
        rows.append(event["photos"].get("1"))
    seen = set()
    for row in rows:
        if not row or not row.get("image"):
            continue
        key = (row["image"], str(row.get("frame_id")), row.get("view"))
        if key not in seen:
            seen.add(key)
            yield row


def bbox_angles(photo, view):
    """Recorded cube/crop camera rays -> the SAME source ERP camera rays."""
    rotation = view.get("face_rotation")
    intrinsic = view.get("K")
    shape = view.get("inference_size")
    if rotation is None or intrinsic is None or shape is None:
        return None
    w, h = float(photo["width"]), float(photo["height"])
    iw, ih = map(float, shape)
    require(min(w, h, iw, ih) > 0, "Invalid comparison image dimensions")
    x0, y0, x1, y1 = map(float, photo["bbox"])
    require(0 <= x0 < x1 <= w and 0 <= y0 < y1 <= h, "Comparison box outside actual photograph")
    t = np.linspace(0, 1, 20)
    pixels = np.concatenate([
        np.c_[x0 + t * (x1 - x0), np.full_like(t, y0)],
        np.c_[np.full_like(t, x1), y0 + t * (y1 - y0)],
        np.c_[x1 - t * (x1 - x0), np.full_like(t, y1)],
        np.c_[np.full_like(t, x0), y1 - t * (y1 - y0)],
    ])
    pixels = (pixels + .5) * np.array([iw / w, ih / h]) - .5
    uv = np.c_[pixels, np.ones(len(pixels))]
    k, rotation = np.asarray(intrinsic, float), np.asarray(rotation, float)
    require(k.shape == rotation.shape == (3, 3), "Invalid recorded cube/crop camera matrix")
    require(np.all(np.isfinite(k)) and np.all(np.isfinite(rotation)), "Non-finite camera metadata")
    rays = uv @ np.linalg.inv(k).T @ rotation.T
    rays /= np.linalg.norm(rays, axis=1, keepdims=True)
    angles = np.degrees(np.c_[np.arctan2(rays[:, 0], rays[:, 2]), -np.arctan2(rays[:, 1], np.hypot(rays[:, 0], rays[:, 2]))])
    return angles.round(6).tolist()


def coordinate_metadata(job):
    units = job["config"].get("paper_integration", {}).get("coordinate_units", "u")
    require(isinstance(units, str) and bool(units.strip()), "Missing configured coordinate units")
    verified = {capture["id"]: job["dependencies"][capture["id"] + ".reconstruct"].get("metric_scale_verified") is True for capture in job["config"]["captures"]}
    return {"coordinate_units": units, "metric_scale_verified": bool(verified) and all(verified.values()), "captures": verified}


def change_summary(detector):
    """Grouping does not resolve uncertain added/removed/identity decisions."""
    events = detector.get("events", [])
    return {"viewer_supported": True, "viewer_kind": "paper_native", "url": "#changes",
            "candidates": len(events), "verified_count": len(detector.get("verified_changes", [])),
            "deferred": sum(event.get("type") == "unresolved" for event in events),
            "ungrouped": sum(not event.get("verified_change") for event in events),
            "verification_kind": "rule_grouping_not_independent_ground_truth"}


def build_changes(manifest, detector, inputs, keypoint_plan, detection_manifest, link_photo):
    """Pure current-plan binding; does not mutate event decisions or infer links."""
    after_id = inputs["after"]["id"]
    captures = {capture["id"]: capture for capture in manifest["captures"]}
    require(after_id in captures, "After capture is absent from current native publication")
    current_views = {view["id"]: view for view in captures[after_id]["views"]}
    anchors = keypoint_plan["keypoints"]
    require(all(row["anchor_capture"] == after_id for row in anchors), "Current change anchors must belong to After")
    anchor_by_id = {row["id"]: row for row in anchors}
    require(len(anchor_by_id) == len(anchors), "Duplicate change anchor IDs")
    require({row["viewpoint_id"] for row in anchors} == set(current_views), "Change anchors differ from freshly selected viewpoints")
    frames = {str(row["id"]): row for row in inputs["after"]["frames"]}
    metadata = {row["id"]: row for row in detection_manifest.get("views", [])}
    groups = copy.deepcopy(detector.get("verified_changes", []))
    group_by_id = {group["id"]: group for group in groups}
    require(len(group_by_id) == len(groups), "Duplicate verified change group IDs")
    events = detector.get("events", [])
    require(len({event["id"] for event in events}) == len(events), "Duplicate source event IDs")
    rows = {view_id: {"viewpoint_id": view_id, "items": [], "evidence_ids": [], "event_ids": [], "verified_change_ids": []} for view_id in current_views}
    evidence = []
    source_events = []
    for event in events:
        group_id = event.get("verified_change")
        require(not group_id or group_id in group_by_id, "Event references unknown verified change group")
        source_events.append({k: event.get(k) for k in ("id", "type", "label", "verified_change", "verification", "reason_code")})
        for photo in after_photos(event):
            frame = frames.get(str(photo.get("frame_id")))
            require(frame is not None, "Detector photograph is not from current After input frames")
            require(photo.get("capture_id", after_id) == after_id, "Evidence photograph capture mismatch")
            frame_index = frame["source_frame_index"]
            require(photo.get("source_frame_index", frame_index) == frame_index, "Evidence source-frame mismatch")
            allowed = set(frame.get("keypoint_ids", []))
            if not allowed:
                allowed = {anchor["id"] for anchor in anchors for side in anchor.get("sides", []) if side["capture"] == after_id and any(str(f["id"]) == str(frame["id"]) for f in side.get("frames", []))}
            require(allowed <= set(anchor_by_id), "Evidence references a stale or unknown change anchor")
            eid = "evidence-" + hashlib.sha256((event["id"] + "|" + photo["image"]).encode()).hexdigest()[:16]
            image = link_photo(photo)
            entry = {"id": eid, "event_id": event["id"], "type": event["type"], "label": event["label"], "verified_change": group_id,
                     "image": image, "bbox": photo.get("bbox"), "width": photo["width"], "height": photo["height"],
                     "capture_id": after_id, "source_frame_index": frame_index, "frame_id": str(frame["id"]),
                     "source_inference_view": photo.get("source_inference_view", photo.get("view")), "role": photo.get("role"),
                     "viewpoint_ids": sorted(anchor_by_id[aid]["viewpoint_id"] for aid in allowed)}
            evidence.append(entry)
            for aid in sorted(allowed):
                viewpoint = anchor_by_id[aid]["viewpoint_id"]
                row = rows[viewpoint]
                row["evidence_ids"].append(eid)
                row["event_ids"].append(event["id"])
                if group_id:
                    row["verified_change_ids"].append(group_id)
                selected = current_views[viewpoint]
                # Proximity/anchor membership never makes two photographs equal.
                if selected.get("source_frame", {}).get("source_frame_index") != frame_index:
                    continue
                source_view = metadata.get(entry["source_inference_view"], {})
                if source_view.get("epoch") != 1 or str(source_view.get("frame_id")) != str(frame["id"]):
                    continue
                angles = bbox_angles(photo, source_view)
                if angles is None:
                    continue
                rays = np.radians(np.asarray(angles))
                direction = np.c_[np.sin(rays[:, 0]) * np.cos(rays[:, 1]), -np.sin(rays[:, 1]), np.cos(rays[:, 0]) * np.cos(rays[:, 1])].mean(0)
                row["items"].append({"id": group_id or event["id"], "source_event_id": event["id"], "event_ids": [event["id"]],
                    "type": event["type"], "label": group_by_id[group_id]["label"] if group_id else event["label"],
                    "verified_change": group_id, "angular_samples": angles,
                    "yaw": float(np.degrees(np.arctan2(direction[0], direction[2]))),
                    "pitch": float(-np.degrees(np.arctan2(direction[1], np.hypot(direction[0], direction[2])))),
                    "evidence_ids": [eid], "display_origin": "same_source_frame_recorded_cube_camera_rays",
                    "frame_binding": {"source_frame_index": frame_index, "input_frame_id": str(frame["id"]), "input_image_sha256": frame["sha256"],
                                      "selected_viewpoint_id": viewpoint, "source_inference_view": entry["source_inference_view"], "current_anchor_id": aid}})
    for row in rows.values():
        for key in ("evidence_ids", "event_ids", "verified_change_ids"):
            row[key] = sorted(set(row[key]))
        merged = {}
        for item in row["items"]:
            if item["id"] not in merged:
                merged[item["id"]] = item
            else:
                prior = merged[item["id"]]
                prior["angular_samples"] += item["angular_samples"]
                prior["event_ids"] = sorted(set(prior["event_ids"] + item["event_ids"]))
                prior["evidence_ids"] = sorted(set(prior["evidence_ids"] + item["evidence_ids"]))
        row["items"] = list(merged.values())
    return {"schema": "paper-native-current-changes-v1", "after_capture": after_id,
            "captures": {cid: {"views": list(rows.values()) if cid == after_id else [{"viewpoint_id": view["id"], "items": [], "evidence_ids": [], "event_ids": [], "verified_change_ids": []} for view in capture["views"]]} for cid, capture in captures.items()},
            "verified_changes": groups, "source_events": source_events, "evidence": evidence,
            "counts": {"source_events": len(events), "verified_changes": len(groups), "evidence_photographs": len(evidence), "same_frame_erp_boxes": sum(len(row["items"]) for row in rows.values())},
            "policy": {"after_only": True, "current_anchor_scope_only": True, "cross_frame_bbox_reuse": False, "unmatched_photo_evidence": "in_viewer_drawer"}}


def viewer(job):
    from . import viewer as native_publisher
    native_job = {**job, "viewer_launch": False, "dependencies": {key: value for key, value in job["dependencies"].items() if key != "pair.changes"}}
    native_publisher.viewer(native_job)
    out = Path(job["output"])
    native_result = read(out / "result.json")
    bundle = Path(native_result["bundle"])
    manifest = read(bundle / "manifest.json")
    coordinates = coordinate_metadata(job)
    manifest["coordinate_units"] = coordinates["coordinate_units"]
    manifest["metric_scale_verified"] = coordinates["metric_scale_verified"]
    for capture in manifest["captures"]:
        capture["coordinate_units"] = coordinates["coordinate_units"]
        capture["metric_scale_verified"] = coordinates["captures"][capture["id"]]
    details = manifest.pop("survey", {})
    manifest["runtime_details"] = {key: details[key] for key in ("ratings", "noise", "timings") if key in details}
    manifest["runtime_details"]["version"] = "SpaceWatch Final 2026.11.10.1"
    change = job["dependencies"].get("pair.changes")
    overlay = None
    if change:
        receipts = {record["path"]: record for path in (Path(job["run_root"]) / "stages").glob("*/complete.json") for record in read(path)["artifacts"]}
        checked = set()

        def trusted(path):
            path = Path(path).resolve()
            key = str(path)
            require(key in receipts, "Current change input has no completed-stage receipt: " + key)
            if key not in checked:
                verify(receipts[key])
                checked.add(key)
            return path

        data_path = trusted(change.get("viewer_data") or change["viewer_manifest"])
        root = data_path.parent
        detector = read(data_path)
        input_path = trusted(change["inputs_manifest"])
        keypoint_path = trusted(change["keypoint_plan"])
        detection_manifest_path = trusted(root / "manifest.json")

        def link_photo(photo):
            source = trusted(root / photo["image"])
            require(root.resolve() in source.parents, "Evidence photograph escapes current detector output")
            if photo.get("image_sha256"):
                require(receipts[str(source)]["sha256"] == photo["image_sha256"], "Detector photograph hash mismatch")
            relative = "current-changes/photos/" + hashlib.sha256(str(source).encode()).hexdigest()[:20] + source.suffix
            target = bundle / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            if not target.exists():
                try:
                    os.link(source, target)
                except OSError:
                    shutil.copyfile(source, target)
            if relative not in manifest["assets"]:
                manifest["assets"].append(relative)
            return "/pipeline-assets/" + relative

        overlay = build_changes(manifest, detector, read(input_path), read(keypoint_path), read(detection_manifest_path), link_photo)
        overlay["provenance"] = {"viewer_data": receipts[str(data_path)], "inputs_manifest": receipts[str(input_path)], "keypoint_plan": receipts[str(keypoint_path)], "detection_manifest": receipts[str(detection_manifest_path)]}
        relative = "current-changes/overlay.json"
        save(bundle / relative, overlay)
        manifest["assets"].append(relative)
        manifest["keypoint_changes"] = {"url": "/pipeline-assets/" + relative, **overlay["counts"]}
        manifest["changes"] = change_summary(detector)
    manifest["change_display_policy"] = {"after_only": True, "source_keypoint_scope_only": True, "same_physical_frame_required_for_erp_boxes": True}
    manifest["bundle_id"] = digest({"native_bundle": manifest["bundle_id"], "current_changes": overlay, "publisher": "paper-native-v1"})
    save(bundle / "manifest.json", manifest)
    return result(out, **{**native_result, "service": None, "changes": manifest.get("changes"), "paper_native_integrated": True,
                          "keypoint_changes": manifest.get("keypoint_changes"), "source_event_count": overlay["counts"]["source_events"] if overlay else 0,
                          "verified_change_count": overlay["counts"]["verified_changes"] if overlay else 0})
