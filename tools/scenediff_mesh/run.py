"""Run SceneDiff's original costs on external meshes and instance masks.

Only pure comparison methods are loaded from the pinned upstream source. Pi3
and SAM inference methods are never imported or invoked. Existing instance IDs
replace SceneDiff's within-sequence object discovery/merging.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
from pathlib import Path
import time

import numpy as np
import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def write(path, data):
    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024**2), b""):
            h.update(chunk)
    return h.hexdigest()


def scatter_mean(src, index, dim=0):
    """Native PyTorch equivalent for the dim=0 reductions used by SceneDiff."""
    assert dim == 0 and index.ndim == 1 and src.shape[0] == len(index)
    size = int(index.max()) + 1 if len(index) else 0
    result = torch.zeros((size, *src.shape[1:]), dtype=src.dtype, device=src.device)
    counts = torch.zeros(size, dtype=src.dtype, device=src.device)
    result.index_add_(0, index, src)
    counts.index_add_(0, index, torch.ones_like(index, dtype=src.dtype))
    return result / counts.clamp_min(1).reshape(-1, *([1] * (src.ndim - 1)))


def load_functions(path, names, namespace, class_name=None):
    """Compile named upstream definitions without loading unused model imports."""
    tree = ast.parse(Path(path).read_text(encoding="utf-8-sig"))
    if class_name:
        tree = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == class_name)
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
    assert {n.name for n in nodes} == set(names), (path, names)
    if class_name:
        nodes = [ast.ClassDef(name=class_name, bases=[], keywords=[], body=nodes, decorator_list=[])]
    module = ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[]))
    exec(compile(module, str(path), "exec"), namespace)


def region_match_cost(a, b, ma, mb, va, vb):
    """Upstream cosine region cost, including empty external segmentation cases."""
    ia, ib = ma.unique(), mb.unique()
    ia, ib = ia[ia >= 0], ib[ib >= 0]
    ca, cb = torch.zeros_like(ma, dtype=torch.float32), torch.zeros_like(mb, dtype=torch.float32)
    if len(ia) and len(ib):
        fa = torch.stack([a[:, ma == i].mean(1) for i in ia])
        fb = torch.stack([b[:, mb == i].mean(1) for i in ib])
        sim = F.normalize(fa, dim=1) @ F.normalize(fb, dim=1).T
        sa, ja = sim.max(1)
        sb, jb = sim.max(0)
    else:
        sa, sb = torch.zeros(len(ia), device=a.device), torch.zeros(len(ib), device=a.device)
        ja, jb = torch.zeros_like(ia), torch.zeros_like(ib)
    for ids, scores, mask, valid, output in [(ia, sa, ma, va, ca), (ib, sb, mb, vb, cb)]:
        for i, score in zip(ids, scores):
            region = mask == i
            if valid[region].float().mean() >= .5:
                output[region] = (1 - score).clamp_min(0)
    return ca, cb, ja, jb, sa, sb


class DinoFeatures:
    def __init__(self, setup, device):
        import timm
        from safetensors.torch import load_file
        self.device, self.cache = device, {}
        self.model = timm.create_model(setup["model"], pretrained=False, num_classes=0, dynamic_img_size=True)
        weights = Path(setup["checkpoint_dir"]) / "model.safetensors"
        self.model.load_state_dict(load_file(str(weights)), strict=True)
        if hasattr(self.model, "rope") and hasattr(self.model.rope, "periods"):
            self.model.rope.periods = self.model.rope.periods.to(torch.bfloat16).to(torch.float32)
        self.model.to(device).eval()
        self.mean = torch.tensor([.485, .456, .406], device=device).view(1, 3, 1, 1)
        self.std = torch.tensor([.229, .224, .225], device=device).view(1, 3, 1, 1)
        self.audit = {"model": setup["model"], "checkpoint_sha256": sha(weights), "checkpoint_bytes": weights.stat().st_size, "feature_dim": self.model.num_features, "implementation": "timm conversion of DINOv3 ViT-H+/16; original bfloat16 RoPE periods restored", "weights_loaded_strict": True}

    def extract_features(self, rgb):
        key = hashlib.sha256(np.ascontiguousarray(rgb).tobytes()).hexdigest()
        if key not in self.cache:
            x = torch.from_numpy(np.ascontiguousarray(rgb)).permute(2, 0, 1)[None].float().to(self.device)
            # External SceneDiff images are in [0,1], like its original loader.
            if float(x.max()) > 1.01:
                x /= 255
            with torch.inference_mode():
                tokens = self.model.forward_features((x - self.mean) / self.std)
                tokens = tokens[:, self.model.num_prefix_tokens:]
                height, width = x.shape[-2:]
                features = tokens.transpose(1, 2).reshape(1, -1, height // 16, width // 16)
                features = F.interpolate(features, size=(height, width), mode="bilinear", align_corners=False)[0]
            self.cache[key] = features
        return self.cache[key]


def build_pipeline(upstream, semantic, config, device):
    namespace = {"np": np, "torch": torch, "F": F, "math": math, "scatter_mean": scatter_mean}
    utility = ["get_img_coor", "get_valid_points", "reprojected_feature", "valid_mask_after_proj", "foreground_distance", "calculate_mask_percentage", "compute_conf_norm_distance", "ravel_hash_vec", "voxelization", "threshold_maximum_entropy"]
    load_functions(upstream / "utils.py", utility, namespace)
    namespace["get_dino_matched_region_cost"] = region_match_cost
    load_functions(upstream / "modules/geometry_model.py", ["compute_geometric_costs"], namespace, "GeometryModel")
    load_functions(upstream / "modules/mask_model.py", ["aggregate_to_regions"], namespace, "MaskModel")
    methods = ["compute_rendering_cost", "_compute_cost_maps_for_image", "_compute_cluster_costs_single", "_compute_changed_region_threshold"]
    load_functions(upstream / "modules/scenediff.py", methods, namespace, "SceneDiff")
    pipeline = namespace["SceneDiff"]()
    pipeline.config, pipeline.device = config, device
    pipeline.geometry_model = namespace["GeometryModel"]()
    pipeline.mask_model = namespace["MaskModel"]()
    pipeline.semantic_model = semantic
    return pipeline, namespace


def voxel_costs(points, costs, visible, voxel_size, namespace, device):
    p = torch.from_numpy(points.reshape(-1, 3)).to(device)
    indices = namespace["voxelization"](p[None], voxel_size)[0]
    mask = visible.flatten()
    means = scatter_mean(costs.flatten()[mask], indices[mask])
    full = torch.zeros(int(indices.max()) + 1, device=device)
    full[:len(means)] = means
    return full[indices]


def threshold(pipeline, costs, visible):
    values = costs[visible]
    if not values.numel() or float(values.max() - values.min()) < 1e-5:
        return float(values.max()) + 1e-5 if values.numel() else 1e6
    q = torch.quantile(values, .7)
    upper = values[values > q]
    if len(upper) < 2 or float(upper.max() - upper.min()) < 1e-5:
        return float(values.max()) + 1e-5
    return float(pipeline._compute_changed_region_threshold(costs, visible))


def process_zone(folder, semantic, config, upstream, device, control=False):
    meta, data = read(folder / "manifest.json"), dict(np.load(folder / "inputs.npz"))
    views = meta["views"]
    if control:
        views = [dict(views[0], index=0), dict(views[0], index=0, epoch=1)]
    images_np = np.stack([data[f"rgb_{v['index']}"] for v in views]) / np.float32(255)
    images = torch.from_numpy(images_np).permute(0, 3, 1, 2)[None].to(device)
    points = np.stack([data[f"points_{v['index']}"] for v in views])
    depth = np.stack([data[f"depth_{v['index']}"] for v in views])
    valid = torch.from_numpy(np.stack([data[f"valid_{v['index']}"] for v in views])).to(device)
    poses = np.stack([data[f"pose_{v['index']}"] for v in views])
    intrinsic = torch.from_numpy(np.stack([data[f"K_{v['index']}"] for v in views]))[None].to(device)
    masks = [data[f"mask_{v['index']}"] for v in views]
    # Same scale normalization as SceneDiff, computed only over observed geometry.
    valid_points = points[valid.cpu().numpy()]
    span = np.percentile(valid_points, 99, axis=0) - np.percentile(valid_points, 1, axis=0)
    scale = 1 / float(span.max())
    points *= scale
    depth *= scale
    poses[:, :3, 3] *= scale
    depth = torch.from_numpy(depth)[None, ..., None].to(device)
    poses = torch.from_numpy(poses)[None].to(device)
    size = meta["size"]
    grid = torch.tensor([[2 / (size - 1), 0, -1], [0, 2 / (size - 1), -1]], device=device)
    pipeline, namespace = build_pipeline(upstream, semantic, config, device)
    xy = namespace["get_img_coor"](size, size).to(device)
    weights = torch.zeros(len(views), len(views), device=device)
    overlap = torch.zeros_like(weights)
    for i, a in enumerate(views):
        for j, b in enumerate(views):
            if a["epoch"] == b["epoch"]:
                continue
            one = namespace["calculate_mask_percentage"](poses[0, i], poses[0, j], depth[0, i].permute(2, 0, 1), depth[0, j].permute(2, 0, 1), intrinsic[0, i], intrinsic[0, j], grid, .3, xy, size, size)
            overlap[i, j] = one
        good = overlap[i] > config["processing"]["visible_percentage"]
        if good.any():
            weights[i, good] = overlap[i, good]
        elif float(overlap[i].max()) > .02:
            weights[i, overlap[i].argmax()] = 1
        if weights[i].sum() > 0:
            weights[i] /= weights[i].sum()
    confidence = [[valid[i].float()] for i in range(len(views))]
    channels, visible = [], []
    for i in range(len(views)):
        if weights[i].sum() == 0:
            channels.append(torch.zeros(3, size, size))
            visible.append(torch.zeros(size, size, dtype=torch.bool))
        else:
            cost, seen = pipeline._compute_cost_maps_for_image(i, images, masks, poses, depth, intrinsic, confidence, weights, valid, xy, grid, size, size)
            assert torch.isfinite(cost).all(), (folder.name, i)
            channels.append(cost)
            visible.append(seen)
    channels = torch.stack(channels).to(device)
    visible = torch.stack(visible).to(device)
    target = torch.from_numpy(np.stack(masks) >= 0).to(device)
    visible &= target & valid
    combined = channels[:, 0] + .1 * channels[:, 1] + .5 * channels[:, 2]
    n0 = sum(v["epoch"] == 0 for v in views)
    cluster = {}
    for name, raw_cost in [("geometry", channels[:, 0]), ("combined", combined)]:
        maps = []
        for first, end in [(0, n0), (n0, len(views))]:
            per_point = voxel_costs(points[first:end], raw_cost[first:end], visible[first:end], 1/200, namespace, device)
            maps.append(pipeline._compute_cluster_costs_single(per_point, masks[first:end], visible[first:end], end-first, size, size))
        cluster[name] = torch.cat(maps)
    thresholds = {k: threshold(pipeline, v, visible) for k, v in cluster.items()}
    rows = {}
    for i, view in enumerate(views):
        mask = torch.from_numpy(masks[i]).to(device)
        feat = semantic.extract_features(images_np[i])
        for local, oid in view["region_to_input_instance"].items():
            region = mask == int(local)
            seen = region & visible[i]
            coverage = float(seen.sum() / region.sum().clamp_min(1))
            support = int(seen.sum()) >= 16 and coverage > .3
            row = rows.setdefault(oid, {"id": oid, "epoch": view["epoch"], "zone": meta["zone"], "views": [], "_features": []})
            evidence = {"index": i, "prepared_index": view["index"], "map_frame": view["map_frame"], "yaw_degrees": view["yaw_degrees"], "pixels": int(region.sum()), "visible_pixels": int(seen.sum()), "visible_fraction": coverage, "supported": support, "geometry_score": float(cluster["geometry"][i][region].mean()), "combined_score": float(cluster["combined"][i][region].mean()), "appearance_score": float(channels[i, 2][region].mean()), "region_match_score": float(channels[i, 1][region].mean())}
            row["views"].append(evidence)
            row["_features"].append(F.normalize(feat[:, region].mean(1), dim=0).cpu().numpy())
    features = {}
    for row in rows.values():
        supported = [v for v in row["views"] if v["supported"]]
        row["supported_views"] = len(supported)
        row["status"] = "compared" if len(supported) >= 2 else "insufficient_observation"
        for name in ["geometry", "combined", "appearance", "region_match"]:
            row[name + "_score"] = float(np.mean([v[name + "_score"] for v in supported])) if supported else None
        for name in ["geometry", "combined"]:
            row[name + "_candidate"] = bool(len(supported) >= 2 and row[name + "_score"] > thresholds[name])
        desc = np.mean(row.pop("_features"), 0)
        features[row["id"]] = desc / max(float(np.linalg.norm(desc)), 1e-8)
    audit = {"zone": meta["zone"], "thresholds": thresholds, "normalization_scale": scale, "normalized_map_error": meta["map_error_u"]*scale, "overlap_matrix": overlap.cpu().tolist(), "weights": weights.cpu().tolist(), "valid_compared_pixels": int(visible.sum()), "source_views": views, "max_abs_geometric_cost": float(channels[:,0].abs().max()), "max_abs_rgb_cost": float(channels[:,2].abs().max()), "control": control}
    if not control:
        np.savez_compressed(folder / "costs.npz", channels=channels.cpu().numpy(), visible=visible.cpu().numpy(), geometry=cluster["geometry"].cpu().numpy(), combined=cluster["combined"].cpu().numpy())
    semantic.cache.clear()
    torch.cuda.empty_cache()
    return list(rows.values()), features, audit


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--upstream", type=Path, required=True)
    parser.add_argument("--setup", type=Path, required=True)
    args = parser.parse_args()
    started = time.time()
    root = args.root
    def state(stage, **kwargs):
        write(root / "run-status.json", {"stage": stage, "seconds": time.time()-started, **kwargs})
    try:
        import yaml
        torch.set_num_threads(4)
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        state("loading_dinov3")
        setup = read(args.setup)
        config = yaml.safe_load((args.upstream / "configs/scenediff_config.yml").read_text())
        semantic = DinoFeatures(setup, "cuda")
        write(root / "model.json", semantic.audit)
        all_rows, all_features, audits = [], {}, []
        manifest = read(root / "prepared/manifest.json")
        ready = [z for z in manifest["zones"] if z["state"] == "ready"]
        assert ready, "No common region with at least two views per side"
        _, _, control = process_zone(root / "prepared" / ready[0]["zone"], semantic, config, args.upstream, "cuda", control=True)
        control["passed"] = control["max_abs_geometric_cost"] < 1e-4 and control["max_abs_rgb_cost"] < 1e-4
        write(root / "self-comparison-control.json", control)
        assert control["passed"], control
        for zone in ready:
            state("comparing", zone=zone["zone"], completed=len(audits), total=len(ready))
            rows, features, audit = process_zone(root / "prepared" / zone["zone"], semantic, config, args.upstream, "cuda")
            all_rows.extend(rows)
            all_features.update(features)
            audits.append(audit)
            write(root / f"prepared/{zone['zone']}/result.json", {"objects": rows, "audit": audit})
            print(zone["zone"], "objects", len(rows), "candidates", sum(x["combined_candidate"] for x in rows), flush=True)
        # Preserve every supplied instance, including empty or unobserved ones.
        originals = read(root / "prepared/objects.json")
        lookup = {r["id"]: r for r in all_rows}
        for obj in originals:
            if obj["id"] in lookup:
                lookup[obj["id"]].update({k:v for k,v in obj.items() if k not in ["zone", "epoch", "id"]})
            else:
                row = {**obj, "status": "no_mesh" if obj["faces"] == 0 else "insufficient_counterpart_views", "views": [], "supported_views": 0, "geometry_score": None, "combined_score": None, "geometry_candidate": False, "combined_candidate": False}
                all_rows.append(row)
                lookup[obj["id"]] = row
        matches = []
        for zone in ready:
            for label in sorted({r["label"] for r in all_rows}):
                aa = [r for r in all_rows if r["epoch"] == 0 and r["zone"] == zone["zone"] and r["label"] == label and r["id"] in all_features]
                bb = [r for r in all_rows if r["epoch"] == 1 and r["zone"] == zone["zone"] and r["label"] == label and r["id"] in all_features]
                if not aa or not bb:
                    continue
                sim = np.array([[all_features[a["id"]] @ all_features[b["id"]] for b in bb] for a in aa])
                for i, j in zip(*linear_sum_assignment(-sim)):
                    alternatives = np.r_[np.delete(sim[i], j), np.delete(sim[:, j], i)]
                    margin = float(sim[i,j] - alternatives.max()) if alternatives.size else 1.
                    matches.append({"reference": aa[i]["id"], "source": bb[j]["id"], "similarity": float(sim[i,j]), "margin": margin, "supported": bool(sim[i,j] >= .7 and margin >= .05), "scope": "same-label, same-region assignment hypothesis; physical identity and motion not proven"})
        counts = {"input_instances": len(all_rows), "compared": sum(r["status"] == "compared" for r in all_rows), "geometry_candidates": sum(r["geometry_candidate"] for r in all_rows), "combined_candidates": sum(r["combined_candidate"] for r in all_rows), "rgb_promoted": sum(r["combined_candidate"] and not r["geometry_candidate"] for r in all_rows), "rgb_demoted": sum(r["geometry_candidate"] and not r["combined_candidate"] for r in all_rows)}
        result = {"objects": all_rows, "matches": matches, "zones": audits, "counts": counts, "model": semantic.audit, "pi3_inference": False, "sam_inference": False, "input_geometry_changed": False, "upstream_commit": "62dcbd285fcfe984bb3db55dc01bdcd1a0055714", "reused_methods": ["geometry.compute_geometric_costs", "SceneDiff.compute_rendering_cost", "SceneDiff._compute_cost_maps_for_image", "SceneDiff._compute_cluster_costs_single", "SceneDiff._compute_changed_region_threshold"], "adaptations": ["external mesh depth/point maps and camera poses replace Pi3", "existing SAM3.1 mesh face IDs replace SAM and within-sequence merging", "observed mesh support replaces Pi3 confidence", "timm DINOv3 ViT-H+ converted weights with original RoPE period precision", "native torch scatter mean; equivalent vectorized voxel cost fill", "correct align_corners=True pixel-grid normalization", "external empty/non-contiguous region IDs and zero-overlap views handled", "same-label cross-capture assignment hypotheses reported separately"], "scope": manifest["scope"], "elapsed_seconds": time.time()-started}
        write(root / "results.json", result)
        state("complete", counts=counts)
    except Exception as exc:
        state("failed", error=repr(exc))
        raise


if __name__ == "__main__":
    main()
