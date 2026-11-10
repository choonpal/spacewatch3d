"""Pool original-image object descriptors from the same frozen SimSaC backbone."""
from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np

from .core import read, sha, write


def extract(root, source, checkpoint, device="cuda"):
    import torch
    import torch.nn.functional as F
    from .model import SimSaC
    root = Path(root)
    out = root / "results"
    model = SimSaC(source, checkpoint, device)
    if model.audit["checkpoint_sha256"] != read(out / "model.json")["checkpoint_sha256"]:
        raise ValueError("Object correspondence must use the same SimSaC checkpoint")
    manifest, records = read(root / "manifest.json"), defaultdict(list)
    for number, meta in enumerate(manifest["views"]):
        path = root / "views" / (meta["id"] + ".npz")
        if sha(path) != meta["geometry_sha256"]:
            raise ValueError("Prepared geometry changed")
        view = dict(np.load(path))
        mask_path = out / "views" / (meta["id"] + ".npz")
        person = np.load(mask_path)["person"]
        valid = view["valid"] & ~person & view.get('segmentation_supported', True)
        ids = [int(i) for i in np.unique(view["instance"][valid]) if i > 0]
        masks = [(i, valid & (view["instance"] == i)) for i in ids]
        masks = [(i, mask) for i, mask in masks if mask.sum() >= manifest["settings"]["min_object_pixels"]]
        if not masks:
            continue
        full, _ = model.prep(view["rgb"])
        with torch.inference_mode():
            pyramid = model.model.pyramid(full, eigth_resolution=True)[-2:]
            for iid, mask in masks:
                weights = torch.from_numpy(mask.astype("f4"))[None, None].to(model.device)
                descriptor = []
                for feature in pyramid:
                    weight = F.interpolate(weights, size=feature.shape[-2:], mode="area")
                    pooled = (feature * weight).sum((2, 3)) / weight.sum((2, 3)).clamp_min(1e-8)
                    descriptor.append(F.normalize(pooled, dim=1)[0].cpu().numpy())
                vector = np.concatenate(descriptor)
                vector /= max(np.linalg.norm(vector), 1e-8)
                y, x = np.nonzero(mask)
                records[f'E{meta["epoch"]}-I{iid}'].append(dict(view=meta["id"], frame_id=meta["frame_id"],
                       pixels=int(mask.sum()), bbox=[int(x.min()), int(y.min()), int(x.max()+1), int(y.max()+1)],
                       segmentation_coverage=float(mask.sum()/max(1,int((view['segmentation_instance']==iid).sum()))) if 'segmentation_instance' in view else None,
                       descriptor=vector))
        print("APPEARANCE", number+1, len(manifest["views"]), flush=True)
    arrays, objects = {}, {}
    for key, observations in records.items():
        frames = defaultdict(list)
        for obs in observations:
            frames[obs["frame_id"]].append(obs)
        descriptors = []
        for frame, own in sorted(frames.items()):
            # The best supported face represents one original camera frame.
            eligible=[v for v in own if v['segmentation_coverage'] is None or v['segmentation_coverage'] >= .35]
            if eligible:
                best = max(eligible, key=lambda v: v["pixels"])
                descriptors.append(best["descriptor"])
        if descriptors:
            arrays[key] = np.stack(descriptors).astype("f4")
        objects[key] = dict(frames=sorted(frames), observations=[{k: v for k, v in o.items() if k != "descriptor"} for o in observations])
    np.savez_compressed(out / "object-appearance.npz", **arrays)
    write(out / "object-appearance.json", dict(schema="mesh-simsac-appearance-v1", objects=objects,
          model=model.audit, source_manifest_sha256=sha(root / "manifest.json"),
          descriptors_sha256=sha(out / "object-appearance.npz"),
          method="Masked means of SimSaC VGG quarter/eighth features; one view per original frame; no new learned identity head"))
    return len(objects)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--simsac-source", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    print("objects", extract(args.root, args.simsac_source, args.checkpoint, args.device))
