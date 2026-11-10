"""Original v2 upstream producers + original v5.1.9 statistical preparation.

Raw-video mode executes the accepted extraction/person-mask/Pano2 producers
with explicit vocabulary and coordinate policy. It also supports prepared
bundles. Existing runtimes/models and an original context plan are required.
This is not a universal mask/scale estimator. No virtual plan or Gaussian stage
is executed. Verified reuse is separately labelled; cold execution is unverified.
"""
from __future__ import annotations

import argparse
import ast
import contextlib
import hashlib
import json
import os
from pathlib import Path
import signal
import shutil
import subprocess
import sys
import time
import struct
import zipfile

ROOT = Path(__file__).resolve().parents[1]
PRODUCERS = ROOT / "upstream519_original"
ACTIONS = ("inputs", "infer", "person_masks", "pano", "coordinates", "reconstruct", "mesh", "semantic", "scene")


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".partial")
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    temp.replace(path)


def record(path):
    p = Path(path).resolve()
    return {"path": str(p), "bytes": p.stat().st_size, "sha256": sha(p)}


def verify(item):
    if not isinstance(item, dict) or not item.get("sha256") or not item.get("path"):
        raise ValueError("Every reused input requires an explicit path and SHA256")
    p = Path(item["path"]).resolve()
    if not p.is_file() or sha(p) != item["sha256"] or ("bytes" in item and p.stat().st_size != item["bytes"]):
        raise ValueError("Input missing or changed: " + str(p))
    return p


def npz_array_binding(path, keys):
    """Original contiguous-array digest, streamed without importing NumPy."""
    bindings = {}
    with zipfile.ZipFile(path) as archive:
        for key in keys:
            with archive.open(key + ".npy") as handle:
                magic = handle.read(8)
                if magic[:6] != b"\x93NUMPY":
                    raise ValueError("Invalid NPY member")
                size = struct.unpack("<H" if magic[6] == 1 else "<I", handle.read(2 if magic[6] == 1 else 4))[0]
                header = ast.literal_eval(handle.read(size).decode("latin1").strip())
                if header["fortran_order"] or not isinstance(header["descr"], str):
                    raise ValueError("Only original contiguous numeric arrays are supported")
                h = hashlib.sha256(str((header["descr"], header["shape"])).encode())
                for block in iter(lambda: handle.read(1024 * 1024), b""):
                    h.update(block)
                bindings[key] = {"sha256": h.hexdigest(), "dtype": header["descr"], "shape": list(header["shape"])}
    return bindings


def source_snapshot():
    manifest = read(PRODUCERS / "source_manifest.json")
    for row in manifest["files"]:
        p = PRODUCERS / row["relative_path"]
        if not p.is_file() or sha(p) != row["sha256"]:
            raise ValueError("Packaged original producer changed: " + str(p))
    return manifest


def context_fields(plan, capture_id):
    """Only the three original profile anchors; never old selections or V/Q."""
    if plan.get("capture_id") != capture_id:
        raise ValueError("Profile context plan capture identity mismatch")
    cfg = plan["provenance"]["config"]
    height = cfg["camera_height_m"]
    anchor = plan["route_graph"]["grid_origin_xy_m"]
    tile = plan["coverage_report"]["surface_sampling"]["fallback_tiles"]["origin_xy_m"]
    import math
    if not isinstance(height, (int, float)) or not math.isfinite(height) or height <= 0:
        raise ValueError("Context camera height must be positive finite")
    for xy in (anchor, tile):
        if len(xy) != 2 or not all(isinstance(x, (int, float)) and math.isfinite(x) for x in xy):
            raise ValueError("Invalid context grid/tile origin")
    return {"capture_id": capture_id, "provenance": {"config": {"camera_height_m": height}},
            "route_graph": {"grid_origin_xy_m": anchor},
            "coverage_report": {"surface_sampling": {"fallback_tiles": {"origin_xy_m": tile}}}}


def memory_available_gib():
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) / 1024**2
    raise RuntimeError("Cannot measure available RAM")


def _runtime_env(config, action, gpu=None):
    runtime = config["runtime"]
    python = runtime["ooops_python"] if action == "infer" else runtime["reconstruction_python"] if action == "reconstruct" else runtime["geometry_python"]
    if not Path(python).is_file():
        raise ValueError("Required original runtime is unavailable: " + python)
    env = dict(os.environ)
    env.update(PYTHONPATH=str(PRODUCERS / "network_guard") + ":" + str(PRODUCERS / "v2") + ":" + str(PRODUCERS / "v2/vendor/spacewatch3d_src"),
               PYTHONDONTWRITEBYTECODE="1", PYTHONUNBUFFERED="1", OMP_NUM_THREADS="1",
               OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1", NUMEXPR_NUM_THREADS="1",
               CUDA_VISIBLE_DEVICES="" if gpu is None else gpu["uuid"],
               CAPSTONE_GPU_THREAD_ID=runtime.get("gpu_thread_id", os.environ.get("CAPSTONE_GPU_THREAD_ID", "")),
               HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", HF_DATASETS_OFFLINE="1",
               HF_HUB_DISABLE_TELEMETRY="1", WANDB_DISABLED="true", WANDB_MODE="disabled")
    env["TORCH_HOME"] = runtime.get("upstream_torch_home", runtime.get("change_torch_home", os.environ.get("TORCH_HOME", "/root/.cache/torch")))
    if not Path(env["TORCH_HOME"]).is_dir():
        raise ValueError("An existing offline Torch cache directory is required: " + env["TORCH_HOME"])
    env["LD_LIBRARY_PATH"] = runtime.get("geometry_library_path", "/workspace/mesh_generator_v1/.runtime/usr/lib/x86_64-linux-gnu") + ":" + env.get("LD_LIBRARY_PATH", "")
    return python, env


def _one_cpu():
    if hasattr(os, "sched_getaffinity"):
        allowed = os.sched_getaffinity(0)
        os.sched_setaffinity(0, {min(allowed)})


def _execute(command, env, log, timeout):
    """Only this owned process group is terminated on its finite timeout."""
    with Path(log).open("w") as stream:
        process = subprocess.Popen(command, cwd=PRODUCERS / "v2", env=env,
                                   stdout=stream, stderr=subprocess.STDOUT, start_new_session=True,
                                   preexec_fn=_one_cpu)
        try:
            code = process.wait(timeout=timeout)
            if code:
                raise RuntimeError("Original worker failed (exit " + str(code) + "): " + str(log))
        except BaseException:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
            raise


def _capture(config, capture_id):
    matches = [c for c in config["captures"] if c["id"] == capture_id]
    if len(matches) != 1:
        raise ValueError("Producer config must contain exactly one matching capture ID")
    return matches[0]


def _profile_context(request, capture_id):
    path = verify(request["profile_context_plan"])
    return path, context_fields(read(path), capture_id)


def _policy_inputs():
    """The unchanged preparation script reads this accepted external release."""
    manifest = read(PRODUCERS / "source_manifest.json")
    rows = [r for r in manifest["files"] if r.get("role") == "accepted_statistical_policy"]
    for row in rows:
        original = Path(row["source"])
        if not original.is_file() or sha(original) != row["sha256"]:
            raise ValueError("Original profile-template/manual/class-score dependency changed: " + str(original))
    return [{"path": r["source"], "sha256": r["sha256"]} for r in rows]


def preflight(request):
    manifest = source_snapshot()
    capture_id = request["capture_id"]
    if request["mode"] not in ("fresh", "verified_reuse"):
        raise ValueError("mode must be fresh or verified_reuse")
    context, fields = _profile_context(request, capture_id)
    report = {"mode": request["mode"], "capture_id": capture_id,
              "producer_inventory_sha256": sha(PRODUCERS / "source_manifest.json"),
              "producer_files": len(manifest["files"]), "profile_context_plan": record(context),
              "context_fields_only": fields, "old_selected_views_imported": False,
              "old_VQ_imported": False, "fresh_model_inference_executed": False,
              "raw_video_mode_supported": True, "arbitrary_policy_free_raw_video_mode_supported": False}
    if request.get("pipeline_config_template"):
        report["pipeline_config_template"] = record(verify(request["pipeline_config_template"]))
    if request["mode"] == "verified_reuse":
        reused = request["reuse"]
        required = ("semantic_arrays", "vocabulary", "coordinates", "scene", "profile_checksums", "profile_input_binding")
        verified = {k: record(verify(reused[k])) for k in required}
        profile = Path(reused["profile_root"]).resolve()
        if Path(verified["profile_checksums"]["path"]) != profile / "checksums.json":
            raise ValueError("Profile checksum manifest is not bound to profile_root")
        checks = read(profile / "checksums.json")
        for rel, digest in checks.items():
            p = (profile / rel).resolve()
            if profile not in p.parents or not p.is_file() or sha(p) != digest:
                raise ValueError("Accepted profile changed: " + rel)
        scene = read(verified["scene"]["path"])
        binding = read(verified["profile_input_binding"]["path"])
        if scene["capture_id"] != capture_id or binding["capture_id"] != capture_id:
            raise ValueError("Reused scene/profile capture mismatch")
        if binding["coordinates_sha256"] != verified["coordinates"]["sha256"]:
            raise ValueError("Reused profile coordinate identity mismatch")
        if Path(binding["source_arrays"]).resolve() != Path(verified["semantic_arrays"]["path"]):
            raise ValueError("Reused profile semantic source path mismatch")
        if Path(binding["source_run"]).resolve() != Path(reused["source_run"]).resolve():
            raise ValueError("Reused profile reconstruction source mismatch")
        arrays = npz_array_binding(verified["semantic_arrays"]["path"], ("vertices", "faces", "rgb", "face_class_id"))
        if any(arrays[k]["sha256"] != value for k, value in binding["arrays"].items()):
            raise ValueError("Reused semantic arrays do not match original profile input binding")
        source = Path(reused["source_run"]).resolve()
        required_files = {"status": "status.json", "config": "config.json", "poses": "result/camera_poses.json", "manifest": "input/manifest.json"}
        source_records = {}
        for key, relative in required_files.items():
            p = verify(reused["reconstruction_files"][key])
            if p != source / relative:
                raise ValueError("Reconstruction file binding mismatch: " + key)
            source_records[key] = record(p)
        if read(source / "status.json").get("status") != "complete":
            raise ValueError("Reused reconstruction is not complete")
        if read(source / "config.json").get("generator_version") != "2.0.0":
            raise ValueError("Reused reconstruction is not original Pano stable v2")
        if len(read(source / "input/manifest.json")["frames"]) != len(read(source / "result/camera_poses.json")):
            raise ValueError("Reused source manifest/pose counts differ")
        if reused.get("captured_prediction"):
            prediction = reused["captured_prediction"]
            infer_path = verify(prediction["inference"])
            vocab_path = verify(prediction["vocabulary"])
            if sha(vocab_path) != verified["vocabulary"]["sha256"]:
                raise ValueError("Reused prediction vocabulary differs from semantic source")
            inference = read(infer_path)
            frames = read(source / "input/manifest.json")["frames"]
            if len(inference["frames"]) != len(frames):
                raise ValueError("Reused prediction frame count differs")
            for i, (row, frame) in enumerate(zip(inference["frames"], frames)):
                if row["source_sha256"] != frame["sha256"] or row["prediction_sha256"] != sha(Path(prediction["predictions"]) / f"{i:05d}.npz"):
                    raise ValueError("Reused prediction is not bound to source pixels")
            report["captured_prediction"] = prediction
        report.update(verified_artifacts=verified, profile_root=str(profile),
                      source_run=str(Path(reused["source_run"]).resolve()),
                      reconstruction_files=source_records, semantic_array_bindings=arrays,
                      manual_and_importance_cache={k: checks[k] for k in ("manual.json", "prompt.txt", "importance_cache.json")},
                      semantic_producer_provenance=request.get("semantic_producer_provenance", "explicit artifact reuse; no new semantic inference"),
                      profile_checksum_files=len(checks), status="verified_reuse_ready")
        return report
    config_path = verify(request["producer_config"])
    config = read(config_path)
    if config.get("pipeline_version") != "2.0.0" or config.get("schema_version") != 1:
        raise ValueError("Fresh upstream requires original v2 producer config")
    cap = _capture(config, capture_id)
    if not cap.get("reconstruction_config") or any(cap.get(k) for k in ("source_run", "predictions", "mesh_arrays", "semantic_arrays", "planning_scene")):
        raise ValueError("Fresh mode requires reconstruction_config and no cached geometry/OOOPS/scene substitutions")
    input_mode = request.get("input_mode", "prepared_bundle")
    if input_mode not in ("prepared_bundle", "original_raw_video"):
        raise ValueError("input_mode must be original_raw_video or prepared_bundle")
    vocabulary = verify(request["vocabulary"])
    if vocabulary != Path(config["vocabulary"]).resolve():
        raise ValueError("Vocabulary config binding mismatch")
    catalog = read(vocabulary)
    byname = {r["name"]: r["id"] for r in catalog}
    if byname.get("floor") != 1 or byname.get("ceiling") != 2 or "person" not in byname:
        raise ValueError("Original coordinate producer requires floor1/ceiling2 and a unique person class")
    report["class_binding"] = {"person": byname["person"], "floor": 1, "ceiling": 2,
                               "original_raw_person_id": 16, "comparison_nodes_changed": 2 if byname["person"] != 16 else 0}
    for key, item in request.get("manual_inputs", {}).items():
        verify(item)
        expected = PRODUCERS / "statistical_policy_snapshot/template" / ("manual.json" if key == "manual" else "prompt.txt")
        if sha(expected) != item["sha256"]:
            raise ValueError("Manual/prompt differs from accepted class-score cache; new model/API scoring is not authorized")
    if input_mode == "original_raw_video":
        video = verify(request["video"])
        reconstruction = PRODUCERS / "raw_input_producer/reconstruction_template.json"
        rc = read(reconstruction)
        report["inputs"] = {"producer_config": record(config_path), "video": record(video), "vocabulary": record(vocabulary),
                            "reconstruction_template": record(reconstruction)}
        report["input_mode"] = input_mode
        python, probe_env = _runtime_env(config, "reconstruct")
        probe = subprocess.run([python, "-B", "-c",
            "import cv2,json,sys; c=cv2.VideoCapture(sys.argv[1]); print(json.dumps(dict(width=c.get(3),height=c.get(4),fps=c.get(cv2.CAP_PROP_FPS),frames=c.get(cv2.CAP_PROP_FRAME_COUNT)))); c.release()", str(video)],
            env=probe_env, capture_output=True, text=True, check=True, timeout=30, preexec_fn=_one_cpu)
        shape = json.loads(probe.stdout)
        if shape["fps"] <= 0 or shape["width"] != 2*shape["height"] or shape["frames"] < 151:
            raise ValueError("Original raw producer requires a sufficiently long 2:1 ERP video with positive FPS")
        report["raw_video_metadata"] = shape
    else:
        reconstruction = verify(request["reconstruction_config"])
        if reconstruction != Path(cap["reconstruction_config"]).resolve():
            raise ValueError("Reconstruction config binding mismatch")
        rc = read(reconstruction)
    # Run the exact original validator without importing model workers.
    import runpy
    common = runpy.run_path(str(PRODUCERS / "v2/vendor/panovggt_pipeline_v2/common.py"))
    common["validate_config"](rc)
    if input_mode == "prepared_bundle":
        bundle_path = verify(request["input_bundle"])
        if bundle_path != Path(rc["input"]["bundle"]).resolve():
            raise ValueError("Frame/base-mask bundle binding mismatch")
        bundle = read(bundle_path)
        if bundle["source"]["sha256"] != rc["input"]["source_sha256"]:
            raise ValueError("Video/reconstruction source mismatch")
        for item in (bundle["source"], bundle["frame_manifest"], *bundle["frames"].values(), *bundle["base_masks"].values()):
            verify(item)
        frame_manifest = read(bundle["frame_manifest"]["path"])
        if len(frame_manifest["frames"]) != rc["input"]["expected_frames"] or len(bundle["base_masks"]) != len(frame_manifest["frames"]):
            raise ValueError("Frame/base-mask count mismatch")
        report["inputs"] = {"producer_config": record(config_path), "reconstruction_config": record(reconstruction),
                            "input_bundle": record(bundle_path), "vocabulary": record(vocabulary)}
        report["frames"] = len(frame_manifest["frames"])
    coordinate_mode = request.get("coordinate_mode", "provided")
    if coordinate_mode == "provided":
        coords = verify(request["coordinates"])
        xyz = read(coords)
        if xyz.get("up") != [0, 0, 1] or "source_to_mesh" not in xyz:
            raise ValueError("Explicit source-to-metric +Z-up coordinates required")
        report["inputs"]["coordinates"] = record(coords)
    elif coordinate_mode != "original_assumed_2p7m":
        raise ValueError("coordinate_mode must be provided or original_assumed_2p7m")
    for action in ("infer", "reconstruct", "scene"):
        _runtime_env(config, action)
    _policy_inputs()
    report.update(status="fresh_preflight_ready_not_executed", input_mode=input_mode, coordinate_mode=coordinate_mode,
                  available_memory_gib=memory_available_gib(),
                  cold_run_memory_guard_gib=request.get("minimum_available_memory_gib", 16.),
                  cold_execution_currently_blocked_by_ram=memory_available_gib() < request.get("minimum_available_memory_gib", 16.),
                  source_contract="Original stride15/1036x518 extraction, original temporal person/nadir mask policy with explicit class-name binding, Pano stable v2; scale is explicit or original unverified2.7m assumption.")
    return report


def _prepare_profile(request, out, config, results):
    capture_id = request["capture_id"]
    original_context, fields = _profile_context(request, capture_id)
    policy = _policy_inputs()
    bridge = out / "profile_bridge"
    bridge.mkdir()
    save(bridge / "context_plan.json", fields)
    bridged = {capture_id + "." + k: results[capture_id + "." + k] for k in ("semantic", "scene", "reconstruct")}
    bridged[capture_id + ".plan"] = {"plan": str(bridge / "context_plan.json"), "purpose": "original context anchors only"}
    save(bridge / "results.json", bridged)
    save(bridge / "identity.json", {"config": config})
    save(bridge / "binding.json", {"new_upstream_results": bridged, "original_context_plan": record(original_context),
                                  "only_extracted_fields": fields, "selected_views_and_VQ_imported": False,
                                  "original_statistical_policy_dependencies": policy})
    python, env = _runtime_env(config, "scene")
    command = [python, "-B", str(PRODUCERS / "offline_worker.py"), "profile",
               str(PRODUCERS / "prepare_map_profile.py"), "--capture", capture_id,
               "--baseline-run", str(bridge), "--output", str(out / "prepared_profile")]
    _execute(command, env, out / "profile_prepare.log", request.get("stage_timeout_seconds", 7200))
    profile = out / "prepared_profile/profile"
    return {"path": str(profile), "checksums": record(profile / "checksums.json"),
            "preparation": record(out / "prepared_profile/preparation.json"),
            "context_binding": record(bridge / "binding.json")}


def _captured_prediction(out, source_run, predictions, vocabulary):
    manifest_path = Path(source_run) / "input/manifest.json"
    manifest = read(manifest_path)
    folder = Path(predictions)
    complete = read(folder / "complete.json")
    if complete["vocabulary_sha256"] != sha(vocabulary) or complete.get("strict_load") is not True:
        raise ValueError("Original OOOPS completion/vocabulary/strict-load mismatch")
    rows = []
    for i, frame in enumerate(manifest["frames"]):
        prediction = folder / f"{i:05d}.npz"
        rows.append({"frame": i, "source_frame_index": frame.get("source_frame_index"),
                     "source_sha256": frame["sha256"], "prediction_sha256": sha(prediction)})
    inference = {"frames": rows, "source_manifest_sha256": sha(manifest_path),
                 "vocabulary_sha256": sha(vocabulary), "checkpoint_sha256": complete["checkpoint_sha256"],
                 "strict_load": True, "original_complete": record(folder / "complete.json"),
                 "model": "OOOPS original whole-ERP inference; no new probability or label algorithm",
                 "producer": record(PRODUCERS / "v2/spacewatch_pipeline/ooops.py")}
    path = out / "captured_inference.json"
    save(path, inference)
    return {"predictions": str(folder), "inference": record(path), "vocabulary": record(vocabulary)}


def _run_fresh(request, out, report, gpu):
    if memory_available_gib() < request.get("minimum_available_memory_gib", 16.):
        raise RuntimeError("Insufficient measured RAM for fresh original upstream; cold execution remains unverified")
    config = read(verify(request["producer_config"]))
    config = json.loads(json.dumps(config))
    cap = _capture(config, request["capture_id"])
    cid = cap["id"]
    vocabulary = str(verify(request["vocabulary"]))
    raw = out / "raw_inputs"
    raw.mkdir()
    (raw / "coordinates").mkdir()
    (raw / "configs").mkdir()
    template = raw / "code/v5/vendor/panovggt_pipeline_v2/configs/dorm2.json"
    template.parent.mkdir(parents=True)
    shutil.copyfile(PRODUCERS / "raw_input_producer/reconstruction_template.json", template)
    segmentation_link = raw / "code/v2/vendor/segmentation"
    segmentation_link.parent.mkdir(parents=True)
    segmentation_link.symlink_to(PRODUCERS / "v2/vendor/segmentation", target_is_directory=True)
    results, receipts = {}, {}

    def execute(action, command, env, stage, extra=()):
        started = time.monotonic()
        _execute(command, env, stage / "worker.log", request.get("stage_timeout_seconds", 7200))
        paths = {p.resolve() for p in stage.rglob("*") if p.is_file() and p.name != "worker.log"}
        paths.update(Path(p).resolve() for p in extra)
        receipts[cid + "." + action] = {"action": action, "command": command, "seconds": time.monotonic()-started,
                                        "artifacts": [record(p) for p in sorted(paths)], "execution_kind": "fresh original producer"}
        save(out / "stage_receipts.json", receipts)

    def raw_action(action, **kwargs):
        stage = out / "stages" / (cid + ".raw_" + action)
        stage.mkdir(parents=True)
        payload = {"original_raw_source": str(PRODUCERS / "raw_input_producer/prepare_new.py"),
                   "raw_base": str(raw), "vocabulary": vocabulary, "capture_id": cid, "raw_action": action, **kwargs}
        save(stage / "job.json", payload)
        python, env = _runtime_env(config, "reconstruct" if action == "extract" else "scene")
        execute("raw_" + action, [python, "-B", str(PRODUCERS / "offline_worker.py"), "raw", str(stage / "job.json")], env, stage)

    def original(action):
        stage = out / "stages" / (cid + "." + action)
        stage.mkdir(parents=True)
        needs_gpu = action == "infer"
        if needs_gpu and gpu is None:
            from .gpu import lease
            context = lease(timeout=request.get("gpu_wait_seconds", 30), thread_id=config["runtime"].get("gpu_thread_id"))
        else:
            context = contextlib.nullcontext(gpu if needs_gpu else None)
        with context as leased:
            payload = {"config": config, "capture": cap, "action": action, "dependencies": results,
                       "output": str(stage), "run_root": str(out), "gpu": leased}
            save(stage / "job.json", payload)
            python, env = _runtime_env(config, action, leased)
            execute(action, [python, "-B", str(PRODUCERS / "offline_worker.py"), "stage", str(stage / "job.json")], env, stage)
        results[cid + "." + action] = read(stage / "result.json")
        save(out / "results.json", results)
        return results[cid + "." + action]

    if request.get("input_mode", "prepared_bundle") == "original_raw_video":
        (raw / "source").mkdir()
        (raw / "source" / (cid + ".MP4")).symlink_to(verify(request["video"]))
        raw_action("extract")
        manifest = raw / "inputs" / cid / "manifest.json"
        data = read(manifest)
        results[cid + ".inputs"] = {"manifest": str(manifest), "images": [str(manifest.parent / r["file"]) for r in data["frames"]]}
        original("infer")
        raw_action("masks", predictions=results[cid + ".infer"]["predictions"])
        rc_path = raw / "configs" / (cid + "_reconstruction.json")
    else:
        cap["reconstruction_config"] = str(verify(request["reconstruction_config"]))
        original("inputs")
        original("infer")
        rc_path = Path(cap["reconstruction_config"])
    # The actual accepted Dorm4/5 driver uses Pano's own component filtering;
    # it does not append v2 early_semantics.masks/filter rescue stages.
    rc = read(rc_path)
    pano_stage = out / "stages" / (cid + ".pano")
    pano_stage.mkdir(parents=True)
    source = pano_stage / "reconstruction"
    if gpu is None:
        from .gpu import lease
        context = lease(timeout=request.get("gpu_wait_seconds", 30), thread_id=config["runtime"].get("gpu_thread_id"))
    else:
        context = contextlib.nullcontext(gpu)
    with context as leased:
        rc["execution"]["physical_gpu"] = leased["index"]
        rc["execution"]["cpu_threads"] = 1
        save(pano_stage / "generator.json", rc)
        python, env = _runtime_env(config, "reconstruct", leased)
        execute("pano", [python, "-B", str(PRODUCERS / "offline_worker.py"), "pano",
                         str(PRODUCERS / "raw_input_producer/panovggt_pipeline_v2/run.py"),
                         "--config", str(pano_stage / "generator.json"), "--output", str(source)], env, pano_stage)
    if request.get("coordinate_mode", "provided") == "original_assumed_2p7m":
        raw_action("coordinates", source_run=str(source), predictions=results[cid + ".infer"]["predictions"])
        coordinates = raw / "coordinates" / (cid + ".json")
    else:
        coordinates = verify(request["coordinates"])
    cap.pop("reconstruction_config", None)
    cap.update(source_run=str(source), coordinates=str(coordinates))
    for action in ("reconstruct", "mesh", "semantic", "scene"):
        original(action)
    profile = _prepare_profile(request, out, config, results)
    prediction = _captured_prediction(out, source, results[cid + ".infer"]["predictions"], vocabulary)
    # Explicitly inventory raw stage products not contained in worker folders.
    raw_paths = [p for p in raw.rglob("*") if p.is_file() and not p.is_symlink() and "/source/" not in str(p) and "/code/" not in str(p)]
    save(out / "raw_input_receipts.json", [record(p) for p in sorted(raw_paths)])
    return {"status": "prepared", "execution_kind": "fresh original raw/Pano/mesh/semantic producers + original statistical profile; no final selection",
            "capture_id": cid, "source_run": str(source), "coordinates": str(coordinates),
            "semantic_arrays": results[cid + ".semantic"]["arrays"], "vocabulary": vocabulary,
            "semantic_frame_id": cid + "_mesh", "semantic_coordinates_sha256": sha(coordinates),
            "planning_scene": str(Path(profile["path"]) / "scene.json"),
            "planning_profile": {"path": profile["path"], "sha256": profile["checksums"]["sha256"]},
            "captured_prediction": prediction, "profile_preparation": profile,
            "stage_receipts": record(out / "stage_receipts.json"), "raw_input_receipts": record(out / "raw_input_receipts.json"),
            "fresh_model_inference_executed": True, "virtual_plan_or_gaussian_executed": False, "source_provenance": report}


def run(request, output_dir, gpu=None):
    """Prepare one capture for the native capture DAG; does not run selection."""
    started = time.monotonic()
    report = preflight(request)
    out = Path(output_dir).resolve()
    package = ROOT.parent.resolve()
    if package not in out.parents:
        raise ValueError("Upstream output must stay inside integrated submission root")
    out.mkdir(parents=True, exist_ok=False)
    save(out / "request.json", request)
    save(out / "preflight.json", report)
    try:
        if request["mode"] == "verified_reuse":
            refs = report["verified_artifacts"]
            result = {"status": "prepared", "execution_kind": "verified existing upstream/profile reuse; no model inference",
                      "capture_id": request["capture_id"], "source_run": report["source_run"],
                      "coordinates": refs["coordinates"]["path"], "semantic_arrays": refs["semantic_arrays"]["path"],
                      "vocabulary": refs["vocabulary"]["path"], "planning_scene": refs["scene"]["path"],
                      "semantic_frame_id": request["capture_id"] + "_mesh",
                      "semantic_coordinates_sha256": refs["coordinates"]["sha256"],
                      "planning_profile": {"path": report["profile_root"], "sha256": refs["profile_checksums"]["sha256"]},
                      "new_model_calls": 0, "source_provenance": report}
            if report.get("captured_prediction"):
                result["captured_prediction"] = report["captured_prediction"]
        else:
            result = _run_fresh(request, out, report, gpu)
        result.update(elapsed_seconds=time.monotonic() - started, producer_sha256=sha(__file__), cpu_affinity_limit=1)
        save(out / "preparation_result.json", result)
        return result
    except BaseException as exc:
        save(out / "failure.json", {"status": "failed", "error": str(exc), "elapsed_seconds": time.monotonic() - started})
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare-request", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args(argv)
    request = read(args.prepare_request)
    if args.check_only:
        if ROOT.parent.resolve() not in args.output.resolve().parents:
            raise ValueError("Check-only output must stay inside integrated submission root")
        report = preflight(request)
        save(args.output, report)
    else:
        report = run(request, args.output)
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
