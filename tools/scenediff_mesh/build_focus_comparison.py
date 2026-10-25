"""Project frozen occupancy-evidence decisions beside cached SimSaC/ViewDelta masks.

No decisions, geometry, model probabilities, or thresholds are refitted here.
Only candidate-trigger track faces are positive; event-group context is separate.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np
import open3d as o3d
from PIL import Image
from scipy.ndimage import map_coordinates

from prepare import make_grid, rasterize, read, sha, view_rotation, write
from report_pair_models import overlay


BITS = {"suppressed": 1, "deferred": 2, "group_context": 4, "candidate": 8}
COLORS = {"baseline": np.array([255, 180, 84]),
          "deferred": np.array([151, 174, 197]),
          "group_context": np.array([192, 185, 136]),
          "simsac": np.array([203, 153, 255]),
          "viewdelta": np.array([91, 221, 165]),
          "person": np.array([255, 107, 142])}


def project_mask(mask, rotation, directions):
    """Nearest-neighbour ERP label sampling, with a wrapping horizontal seam."""
    rays = directions @ rotation.T
    rays /= np.linalg.norm(rays, axis=-1, keepdims=True)
    h, w = mask.shape
    u = (np.arctan2(rays[..., 0], rays[..., 2]) / (2 * np.pi) + .5) * w - .5
    v = (np.arcsin(np.clip(rays[..., 1], -1, 1)) / np.pi + .5) * h - .5
    padded = np.pad(mask.astype("u1"), ((0, 0), (1, 1)), mode="wrap")
    return map_coordinates(padded, [np.clip(v, 0, h - 1), np.mod(u, w) + 1],
                           order=0, mode="nearest") > 0


def partitions(flags, valid):
    """Exclusive audit categories; no audit support never means unchanged."""
    candidate = (flags & BITS["candidate"]) != 0
    deferred = ((flags & BITS["deferred"]) != 0) & ~candidate
    context = ((flags & BITS["group_context"]) != 0) & ~candidate & ~deferred
    suppressed = ((flags & BITS["suppressed"]) != 0) & ~candidate & ~deferred & ~context
    return dict(candidate=candidate, deferred=deferred, group_context=context,
                suppressed=suppressed, unaudited=valid & (flags == 0), no_mesh=~valid)


def measure(mask, regions, person, person_available):
    counts = {k: int((mask & v).sum()) for k, v in regions.items()}
    assert sum(counts.values()) == int(mask.sum())
    return dict(pixels=int(mask.sum()), fraction=float(mask.mean()), by_baseline=counts,
                on_person_pixels=int((mask & person).sum()) if person_available else None)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("outputs/scenediff-mesh-rgb-20261024"))
    parser.add_argument("--prior", type=Path, default=Path("outputs/registration-benchmark-20261019/new-meshfirst-20261022"))
    args = parser.parse_args()
    root, prior = args.root, args.prior
    source = root / "model-comparison-20261025"
    out = root / "focus-comparison-20261025"
    images = out / "images"
    images.mkdir(parents=True, exist_ok=True)
    manifest = read(source / "inputs/manifest.json")
    rgb = np.load(source / "inputs/rgb.npz")
    review_path = prior / "automatic-lifecycle-v2/review.json"
    audit_path = prior / "all-object-review-v1/face-audit.json"
    review, audit = read(review_path), read(audit_path)
    ctx = read(prior / "alignment-context.json")
    detections = read(prior / "image-lifecycle-newmesh-v1/detections.json")
    detection_frames = {(f["epoch"], f["frame"]): f for f in detections["frames"]}
    tracks = {t["id"]: t for t in review["tracks"]}
    views = manifest["views"]
    zones = sorted({v["zone"] for v in views})
    prepared = {z: np.load(root / "prepared" / z / "inputs.npz") for z in zones}
    probs = {m: np.load(source / "results" / m / "common_view_probabilities.npz")
             for m in ["simsac", "viewdelta"]}
    models = {}
    for m in probs:
        folder = source / "results" / m
        models[m] = {"audit": read(folder / "model.json"),
                     "status": read(folder / "status.json"),
                     "control": read(folder / "self-control.json")}
        assert models[m]["audit"]["loaded_strict"] and models[m]["status"]["state"] == "complete"

    # Preserve the native original-face index; never render a cropped event mesh.
    scenes, flags, poses = {}, {}, {}
    for epoch in [0, 1]:
        mesh = np.load(prior / f"new-inputs/labeled-mesh-{epoch}.npz")
        scenes[epoch] = o3d.t.geometry.RaycastingScene(nthreads=4)
        scenes[epoch].add_triangles(o3d.core.Tensor(mesh["p"].astype("f4")),
                                    o3d.core.Tensor(mesh["faces"].astype("u4")))
        flags[epoch] = np.zeros(len(mesh["faces"]), "u1")
        poses[epoch] = {p["frame"]: np.asarray(p["camera_to_world"], float)
                        for p in read(prior / f"new-inputs/poses-{epoch}.json")["poses"]}
    records = []
    for rec in review["records"]:
        decision = rec["review"]["decision"]
        ids = [rec[k] for k in ["reference", "source"] if rec.get(k) in audit]
        for key in ids:
            a = audit[key]
            flags[a["epoch"]][a["original_face_ids"]] |= BITS[decision]
        records.append(dict(id=rec["id"], zone=rec["zone"], class_name=rec["class_name"],
                            decision=decision, reason=rec["review"]["reason"], ids=ids))
    groups = []
    for group in review["event_groups"]:
        entries = group["entry_ids"]
        for key in entries:
            assert review["track_decisions"][key]["decision"] == "candidate"
            a = audit[key]
            flags[a["epoch"]][a["original_face_ids"]] |= BITS["candidate"]
        groups.append(dict(id=group["id"], title=group["title"], zone=group["zone"], class_name=group["class_name"],
                           entries=entries, context_ids=group["semantic_context_ids"],
                           reason=group["review"]["reason"],
                           decisions={key: review["track_decisions"][key] for key in entries},
                           evidence={key: dict(epoch=audit[key]["epoch"],
                                              faces=len(audit[key]["original_face_ids"]),
                                              own_frames=sorted({o["frame"] for o in tracks[key]["observations"]}),
                                              vacancy_frames=review["track_decisions"][key]["vacancy_frames"])
                                     for key in entries}))

    _, directions = make_grid(224)
    result = dict(schema="frozen-occupancy-vs-pair-models-v1", views=[], groups=groups,
                  zones=zones, baseline_summary=review["summary"], policy=review["policy"],
                  models=models, threshold=.5, pairing=manifest["pairing"],
                  conditions={"physical_event_ground_truth": False, "different_floors": True,
                              "baseline_observations": "Full previous capture; saved automatic decisions",
                              "pair_model_observations": "20 pilot ERP frames / 30 direction views",
                              "candidate_projection": "First-hit original faces of candidate-trigger tracks only",
                              "unselected_is_unchanged": False})
    validation, arrays, person_sources = [], {}, {}
    for v in views:
        key, epoch, zone = v["key"], v["epoch"], v["zone"]
        frame_rgb = rgb[key]
        assert hashlib.sha256(frame_rgb.tobytes()).hexdigest() == v["rgb_sha256"]
        assert np.array_equal(frame_rgb, prepared[zone][f"rgb_{v['index']}"])
        rot = view_rotation(np.deg2rad(v["yaw"]))
        pose = poses[epoch][v["frame"]].copy()
        pose[:3, :3] = pose[:3, :3] @ rot
        face_ids = np.arange(len(flags[epoch]), dtype="i4")
        depth, faces, valid = rasterize(scenes[epoch], pose, directions, face_ids, ctx["height_u"])
        stored = prepared[zone][f"valid_{v['index']}"]
        overlap = valid & stored
        scale = 1.
        if epoch:
            matrix = np.asarray(ctx["methods"]["structure"]["region_matrices_after_Geo"])[int(zone[1:]) - 1] @ np.asarray(ctx["base_matrix"])
            scale = float(np.cbrt(np.linalg.det(matrix[:3, :3])))
        delta = np.abs(depth[overlap] * scale - prepared[zone][f"depth_{v['index']}"][overlap])
        check = dict(key=key, valid_agreement=float((valid == stored).mean()),
                     depth_error_p99_u=float(np.quantile(delta, .99)),
                     depth_error_max_u=float(delta.max()), rgb_hash_preserved=True)
        assert check["valid_agreement"] > .999, check
        assert check["depth_error_p99_u"] < ctx["height_u"] * 2e-4, check
        validation.append(check)
        pixel_flags = np.where(valid, flags[epoch][faces], 0).astype("u1")
        regions = partitions(pixel_flags, valid)
        arrays[f"flags__{key}"] = pixel_flags
        arrays[f"valid__{key}"] = valid
        arrays[f"faces__{key}"] = np.where(valid, faces, -1).astype("i4")
        person = np.zeros(valid.shape, bool)
        detection = detection_frames.get((epoch, v["frame"]))
        person_paths = [prior.parent / o["mask_url"] for o in detection["objects"]
                        if o["class_name"] == "person"] if detection else []
        person_available = detection is not None and all(p.exists() for p in person_paths)
        for path in person_paths:
            if path.exists():
                mask = np.asarray(Image.open(path).convert("L")) > 127
                person |= project_mask(mask, rot, directions)
                person_sources[str(path)] = sha(path)
        arrays[f"person__{key}"] = person
        entry = dict(v, baseline={k: int(mask.sum()) for k, mask in regions.items()},
                     person=dict(available=person_available, pixels=int(person.sum())),
                     map_error_u=ctx["methods"]["structure"]["errors"]["regions"][zone]["tolerance_u"],
                     records=[], visible_groups=[], stats={})
        for rec in records:
            fs = [i for k in rec["ids"] if audit[k]["epoch"] == epoch
                  for i in audit[k]["original_face_ids"]]
            count = int((valid & np.isin(faces, fs)).sum()) if fs else 0
            if count:
                entry["records"].append(dict(rec, visible_pixels=count))
        for group in groups:
            fs = [i for k in group["entries"] if audit[k]["epoch"] == epoch
                  for i in audit[k]["original_face_ids"]]
            count = int((valid & np.isin(faces, fs)).sum()) if fs else 0
            if count:
                entry["visible_groups"].append(dict(id=group["id"], pixels=count))
        Image.fromarray(frame_rgb).save(images / f"{key}-rgb.png")
        baseline = overlay(frame_rgb, regions["candidate"], COLORS["baseline"])
        Image.fromarray(baseline).save(images / f"{key}-baseline.png")
        with_review = overlay(frame_rgb, regions["deferred"], COLORS["deferred"])
        with_review = overlay(with_review, regions["group_context"], COLORS["group_context"])
        with_review = overlay(with_review, regions["candidate"], COLORS["baseline"])
        Image.fromarray(with_review).save(images / f"{key}-baseline-review.png")
        Image.fromarray(overlay(frame_rgb, person, COLORS["person"])).save(images / f"{key}-person.png")
        for mode in ["view", "erp"]:
            masks = {}
            for model in probs:
                probability = probs[model][mode + "__" + key]
                assert probability.shape == valid.shape and np.isfinite(probability).all()
                masks[model] = probability > .5
                Image.fromarray(overlay(frame_rgb, masks[model], COLORS[model])).save(images / f"{key}-{mode}-{model}.png")
            entry["stats"][mode] = {m: measure(mask, regions, person, person_available)
                                    for m, mask in masks.items()}
            entry["stats"][mode]["model_intersection"] = int((masks["simsac"] & masks["viewdelta"]).sum())
            entry["stats"][mode]["model_disagreement"] = int((masks["simsac"] ^ masks["viewdelta"]).sum())
        result["views"].append(entry)
        print(f"{key}: existing={entry['baseline']['candidate']} px; "
              f"SimSaC={entry['stats']['view']['simsac']['pixels']}; "
              f"ViewDelta={entry['stats']['view']['viewdelta']['pixels']}", flush=True)
    np.savez_compressed(out / "baseline-projections.npz", **arrays)
    write(out / "comparison.json", result)
    source_paths = [review_path, audit_path, prior / "alignment-context.json",
                    prior / "image-lifecycle-newmesh-v1/detections.json", source / "inputs/manifest.json"]
    source_paths += [prior / f"new-inputs/{kind}-{epoch}.{ext}"
                     for epoch in [0, 1] for kind, ext in [("labeled-mesh", "npz"), ("poses", "json")]]
    source_paths += [source / "results" / model / "common_view_probabilities.npz" for model in probs]
    write(out / "validation.json", dict(views=validation, source_hashes={str(p): sha(p) for p in source_paths},
                                       person_mask_hashes=person_sources, bit_flags=BITS,
                                       candidate_face_source="event_groups.entry_ids -> face-audit.original_face_ids",
                                       context_never_promoted_to_candidate=True, model_threshold_fixed=.5))
    for filename in ["focus.html", "focus.js"]:
        shutil.copy2(Path(__file__).with_name(filename), out / ("index.html" if filename.endswith("html") else "focus.js"))
    aggregate = {}
    for mode in ["view", "erp"]:
        aggregate[mode] = {m: dict(pixels=sum(v["stats"][mode][m]["pixels"] for v in result["views"]),
                                    by_baseline={k: sum(v["stats"][mode][m]["by_baseline"][k] for v in result["views"])
                                                 for k in regions}) for m in probs}
    positive_pixels = sum(v["baseline"]["candidate"] for v in result["views"])
    coverage = {mode: {m: aggregate[mode][m]["by_baseline"]["candidate"] for m in probs}
                for mode in aggregate}
    report = ["# 기존 정합오차·관측검증 / SimSaC / ViewDelta", "",
              "기존 occupancy-evidence-v2 자동 판정을 고정하고 원본 면을 동일 RGB 시점에 first-hit 투영했다.",
              "주황색 양성 영역은 후보를 발생시킨 트랙의 면만 포함한다. 그룹에 속한 주변 메시 전체를 양성으로 바꾸지 않았다.",
              "보류, 그룹 문맥, 억제, 판정 연결이 없는 표면, mesh hit 없음은 별도 집계한다. 어느 것도 변화 없음의 정답은 아니다.", "",
              f"기존 전체 13구역의 후보 묶음 {len(groups)}개 중 선택된 {', '.join(zones)} 구역에 "
              f"{sum(g['zone'] in zones for g in groups)}개가 있다. "
              f"30뷰에 보이는 후보 트랙 표면은 중복 관측을 포함하여 {positive_pixels:,} 픽셀이다.",
              "기존 방식은 전체 촬영을 이용한 판정이고 두 모델은 파일럿의 20 ERP 프레임 / 30 방향 뷰를 사용했다. 관측량을 맞춘 성능 실험이 아니다.",
              "Dorm2와 Dorm3는 서로 다른 층이며 물리적 시간 변화 정답이 없다. 후보 픽셀 크기는 물체 수·정확도가 아니다.", "",
              "입력 기본값은 동일 방향 RGB. 새 모델의 모든 비영 대응 예측확률 평균 > 0.5를 그대로 사용했다.",
              "ERP 입력 결과도 유지한다. 자기 입력 비교 후보는 두 모델 모두 0이었다.",
              "정확히 같은 프레임의 기존 person segmentation과 겹침은 참고용이며 후보를 삭제하지 않는다. 검출 없음은 사람 부재 정답이 아니다.", "",
              "## 현재 관찰", "",
              f"- R08의 기존 의자 후보가 보이는 3개 시점의 표면 {positive_pixels:,}px 중, "
              f"동일 방향 RGB 입력 SimSaC는 {coverage['view']['simsac']:,}px와 겹쳤고 ViewDelta는 {coverage['view']['viewdelta']:,}px와 겹쳤다.",
              f"- ERP 전체 입력에서는 같은 표면에 SimSaC {coverage['erp']['simsac']:,}px, ViewDelta {coverage['erp']['viewdelta']:,}px가 겹쳤다. 입력 방식에 따라 반응이 달랐다.",
              "- SimSaC는 R08의 기존 보류 영역도 선택했다. ViewDelta가 선택한 보류 면도 있어 판정 이유와 상대 RGB를 함께 보여준다.",
              "- R07에서는 SimSaC가 촬영자 머리, ViewDelta가 머리 또는 주변 문·벽 영역을 선택한 사례가 있다. 두 모델의 원본 후보를 시설물 변화로 바로 채택하지 않았다.", "",
              "## 후보 픽셀 분해", "", "```json", json.dumps(aggregate, ensure_ascii=False, indent=2), "```", "",
              "validation.json에 입력 해시·깊이 검증, baseline-projections.npz에 원본 면·판정 bit·사람 마스크를 보존했다.",
              "각 뷰 records에는 보이는 기존 객체 조각과 저장된 판정 이유가 있다. 픽셀 합계의 배타적 우선순위는 후보 > 보류 > 그룹 문맥 > 억제 > 미연결 > no hit이다."]
    (out / "REPORT.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(json.dumps(dict(candidate_projection_pixels=positive_pixels, totals=aggregate), indent=2))


if __name__ == "__main__":
    main()
