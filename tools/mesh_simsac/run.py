"""Infer fresh SimSaC candidates and aggregate independently observed 3D evidence."""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import time
from pathlib import Path

import numpy as np
from PIL import Image

from .core import evidence_masks, read, sha, vote_state, voxel_components, write


def people_masks(views, checkpoint, device):
    if checkpoint is None:
        return {v["id"]: v["person"].copy() for v in views}, dict(detector_used=False,
                policy="Only supplied image masks and segmented mesh person IDs; unsegmented people may remain")
    from ultralytics import YOLO
    detector, masks = YOLO(str(checkpoint)), {}
    for view in views:
        rgb = view["rgb"]
        # Ultralytics ndarray input is BGR.
        result = detector.predict(rgb[..., ::-1].copy(), classes=[0], conf=.25, imgsz=640,
                                  device=device, verbose=False)[0]
        mask = view["person"].copy()
        h, w = mask.shape
        if result.masks is not None:
            for item in result.masks.data.cpu().numpy():
                mask |= np.asarray(Image.fromarray(item > .5).resize((w, h), Image.Resampling.NEAREST))
        if result.boxes is not None:
            for x0, y0, x1, y1 in result.boxes.xyxy.cpu().numpy():
                dx, dy = (x1 - x0) * .1, (y1 - y0) * .1
                mask[max(0, int(np.floor(y0-dy))):min(h, int(np.ceil(y1+dy))),
                     max(0, int(np.floor(x0-dx))):min(w, int(np.ceil(x1+dx)))] = True
        masks[view["id"]] = mask
    return masks, dict(detector_used=True, checkpoint_sha256=sha(checkpoint),
                       policy="Original RGB YOLO person segmentation OR box +10% margin, plus supplied masks; tested independently in each camera")


class Evidence:
    def __init__(self, settings):
        self.settings = settings
        self.cells = {}
        self.objects = defaultdict(lambda: defaultdict(lambda: [0, 0]))

    def add(self, view, observed, changed):
        if np.any(changed & ~observed):
            raise ValueError("Changed pixels outside observed support")
        epoch, frame = view["epoch"], view["frame_id"]
        voxel = self.settings["voxel_size_u"]
        for mask, column in [(observed, 0), (changed, 1)]:
            points = view["points"][mask]
            if not len(points):
                continue
            keys = np.floor(points / voxel).astype("i8")
            unique = np.unique(keys, axis=0)
            ids = view["instance"][mask]
            for key in unique:
                lookup = (epoch, *map(int, key))
                if lookup not in self.cells:
                    self.cells[lookup] = [set(), set(), set()]
                self.cells[lookup][column].add(frame)
            if column == 1:
                # Store only IDs with changed support, without requiring labels to match across captures.
                for key, iid in zip(keys, ids):
                    if iid > 0:
                        self.cells[(epoch, *map(int, key))][2].add(int(iid))
            instance, counts = np.unique(ids[ids > 0], return_counts=True)
            for iid, count in zip(instance, counts):
                self.objects[(epoch, int(iid))][frame][column] += int(count)

    def finish(self, object_rows):
        settings = self.settings
        object_output = []
        for obj in object_rows:
            records = self.objects[(obj["epoch"], obj["instance_id"])]
            observed, changed = [], []
            for frame, (support, candidates) in sorted(records.items()):
                if support >= settings["min_object_pixels"]:
                    observed.append(frame)
                    if candidates >= settings["min_object_pixels"] and candidates / support >= settings["object_pixel_ratio"]:
                        changed.append(frame)
            state = vote_state(observed, changed, settings["min_frames"], settings["changed_frame_ratio"])
            object_output.append(dict(**obj, state=state, observed_frames=observed, changed_frames=changed,
                                      frame_pixel_counts={f: dict(observed=v[0], changed=v[1]) for f, v in records.items()},
                                      change_type="undetermined" if state == "change_candidate" else None,
                                      physical_identity_verified=False))
        regions, retained = [], []
        for epoch in [0, 1]:
            keys = []
            for key, (observed, changed, ids) in sorted(self.cells.items()):
                if key[0] == epoch and vote_state(observed, changed, settings["min_frames"], settings["changed_frame_ratio"]) == "change_candidate":
                    keys.append(key[1:])
                    retained.append(dict(epoch=epoch, key=list(key[1:]), observed_frames=len(observed), changed_frames=len(changed)))
            for group in voxel_components(keys):
                q = np.array(group, float)
                bounds = np.stack([q.min(0), q.max(0) + 1]) * settings["voxel_size_u"]
                frames, ids = set(), set()
                for key in group:
                    cell = self.cells[(epoch, *key)]
                    frames.update(cell[1])
                    ids.update(cell[2])
                regions.append(dict(id=f"E{epoch}-C{len(regions)+1:04d}", epoch=epoch, bounds=bounds.tolist(),
                                    voxel_count=len(group), source_frames=sorted(frames), instance_ids=sorted(ids),
                                    state="change_candidate", change_type="undetermined", verified_object_change=False,
                                    localization="Segmented mesh first-hit supported by supplied point cloud"))
        return object_output, regions, retained


def run(root, source, checkpoint, person_checkpoint=None, device="cuda"):
    import torch
    from .model import SimSaC
    root = Path(root).resolve()
    manifest = read(root / "manifest.json")
    if manifest.get("schema") != "mesh-simsac-prepared-v1":
        raise ValueError("Expected prepared mesh-simsac inputs")
    if manifest["gaussian_inputs_used"] or manifest["keypoint_plan_used"] or manifest["existing_change_results_used"]:
        raise ValueError("This pipeline must be independent of Gaussians, keypoint plans and old change results")
    out = root / "results"
    if out.exists():
        raise FileExistsError("Use a new prepared run; inference never mixes predictions from different inputs")
    out.mkdir()
    (out / "pairs").mkdir()
    (out / "views").mkdir()
    started = time.time()
    write(root / "inference-status.json", dict(state="loading"))
    try:
        torch.set_num_threads(4)
        torch.manual_seed(0)
        np.random.seed(0)
        settings, views = manifest["settings"], []
        for meta in manifest["views"]:
            path = root / "views" / (meta["id"] + ".npz")
            if sha(path) != meta["geometry_sha256"]:
                raise ValueError(f"Prepared view changed: {path}")
            view = {**meta, **dict(np.load(path, allow_pickle=False))}
            if hashlib.sha256(view["rgb"].tobytes()).hexdigest() != meta["rgb_sha256"]:
                raise ValueError("RGB integrity mismatch")
            views.append(view)
        lookup = {v["id"]: v for v in views}
        masks, person_audit = people_masks(views, person_checkpoint, device)
        model = SimSaC(source, checkpoint, device)
        write(out / "model.json", dict(**model.audit, person=person_audit))
        evidence, audits, view_results, controls = Evidence(settings), [], [], {}
        # One unmodified self-pair per capture checks that inference did run on the new RGB.
        for epoch in [0, 1]:
            candidates = [v for v in views if v["epoch"] == epoch]
            if candidates:
                v = max(candidates, key=lambda row: row["valid_pixels"])
                prob = model.predict(v["rgb"], v["rgb"])
                controls[str(epoch)] = dict(view=v["id"], candidate_pixels=int((prob > settings["probability_threshold"]).sum()),
                                           max_probability=float(prob.max()))
                np.savez_compressed(out / f"self-{epoch}.npz", probability=prob)
        for number, pair in enumerate(read(root / "pairs.json")):
            query = lookup[pair["query"]]
            support = np.zeros(query["valid"].shape, "u4")
            positive = np.zeros_like(support)
            probabilities = []
            for ref_record in pair["refs"]:
                reference = lookup[ref_record["id"]]
                probability = model.predict(query["rgb"], reference["rgb"])
                masks_pair = evidence_masks(query, reference, probability, masks[query["id"]], masks[reference["id"]],
                                            settings["visibility_tolerance_u"], settings["probability_threshold"])
                support += masks_pair["observable"].astype("u1")
                positive += masks_pair["candidate"].astype("u1")
                probabilities.append(probability)
                key = query["id"] + "--" + reference["id"]
                file = out / "pairs" / (key + ".npz")
                np.savez_compressed(file, probability=probability.astype("f4"), **masks_pair)
                audits.append(dict(id=key, query=query["id"], reference=reference["id"],
                                   query_rgb_sha256=query["rgb_sha256"], reference_rgb_sha256=reference["rgb_sha256"],
                                   path=str(file.relative_to(root)), sha256=sha(file),
                                   pixels={k: int(m.sum()) for k, m in masks_pair.items()}))
            observable = support > 0
            changed = observable & (positive >= np.maximum(1, np.ceil(support * settings["changed_frame_ratio"])))
            evidence.add(query, observable, changed)
            probability = np.mean(probabilities, axis=0).astype("f4") if probabilities else np.zeros(support.shape, "f4")
            np.savez_compressed(out / "views" / (query["id"] + ".npz"), mean_probability=probability,
                                observable=observable, candidate=changed, person=masks[query["id"]],
                                reference_count=support)
            overlay = query["rgb"].copy()
            overlay[changed] = (.35 * overlay[changed] + .65 * np.array([255, 75, 30])).astype("u1")
            overlay[masks[query["id"]]] = (.5 * overlay[masks[query["id"]]] + .5 * np.array([60, 120, 255])).astype("u1")
            Image.fromarray(overlay).save(out / "views" / (query["id"] + ".jpg"), quality=92)
            view_results.append(dict(id=query["id"], epoch=query["epoch"], frame_id=query["frame_id"],
                                     references=len(pair["refs"]), observed_pixels=int(observable.sum()),
                                     candidate_pixels=int(changed.sum()), state=pair["state"]))
            write(root / "inference-status.json", dict(state="running", view=number+1, total=len(views), pairs=len(audits), seconds=time.time()-started))
            print("SIMSAC", query["id"], len(pair["refs"]), int(changed.sum()), flush=True)
        objects, regions, voxels = evidence.finish(read(root / "objects.json"))
        write(out / "changes.json", dict(schema="mesh-simsac-changes-v1", objects=objects, regions=regions,
              candidate_voxels=voxels, views=view_results, self_controls=controls,
              gaussian_inputs_used=False, keypoint_plan_used=False, existing_change_results_used=False,
              source_manifest_sha256=sha(root / "manifest.json"), input_scope=manifest["scope"],
              change_types_inferred=False, physical_identity_verified=False,
              caveat="SimSaC image-change candidates with multi-frame geometric support; no temporal ground truth or moved/added/removed guarantee"))
        write(out / "pair-audit.json", audits)
        write(root / "inference-status.json", dict(state="complete", pairs=len(audits), regions=len(regions),
              object_candidates=sum(o["state"] == "change_candidate" for o in objects), seconds=time.time()-started))
    except Exception as error:
        write(root / "inference-status.json", dict(state="error", error=str(error)))
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--simsac-source", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--person-checkpoint", type=Path)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    run(args.root, args.simsac_source, args.checkpoint, args.person_checkpoint, args.device)
