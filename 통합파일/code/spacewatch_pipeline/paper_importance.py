"""Reconstruct saved five-response class importance for the current scene.

Cache-only stage. It has no API transport, credentials, model imports, or network
entry point. Matching manual/class/model identities are required before reuse.
"""
import hashlib
import json
from pathlib import Path
import re
import statistics

from .io import read, record, result, save

VERSION = "patrol-importance-manual-1to5-v1"
MODEL = "gemini-3.1-flash-lite"
REPORT_SHA256 = "cf24cdaca1c85682d0a641f332da4d2c3f8edfe67787bdadafe330e4a619e8ae"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def identity_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def manual_rules(value):
    """The saved protocol's whitespace/numbered-line preprocessing, unchanged."""
    if isinstance(value, dict):
        lines = list(value.values())
    elif isinstance(value, list):
        lines = value
    elif isinstance(value, str):
        lines = [value]
    else:
        raise ValueError("Current patrol manual must be rule strings")
    require(lines and all(isinstance(line, str) for line in lines), "Current patrol manual must be nonempty rule strings")
    rules = []
    for line in "\n".join(lines).splitlines():
        line = re.sub(r"^\s*(?:[-*•]\s+|\d+[.)]\s+|M\d+\s*[:：]\s*)", "", line)
        line = re.sub(r"^\s*(?:장소별 운영 규칙|순찰 매뉴얼|순찰 기준)(?:\s*\([^)]*\))?\s*[:：]\s*", "", line)
        rules.extend(" ".join(part.split()) for part in re.split(r"(?<=[.!?])\s+", line) if part.strip())
    require(bool(rules), "Current patrol manual has no rules")
    return rules


def class_names(value):
    require(isinstance(value, list) and value and all(isinstance(name, str) and name.strip() == name and name for name in value), "Current object classes must be explicit nonempty strings")
    require(len(value) == len(set(value)), "Duplicate current requested class")
    return sorted(value)


def verify_cached_ratings(cache_root, manual, classes, model=MODEL, report_sha256=REPORT_SHA256):
    """Read only the importance JSON tree and reconstruct every class summary."""
    root = Path(cache_root).resolve()
    sources = []

    def source(relative):
        path = root / relative
        require(path.is_file() and not path.is_symlink(), "Missing regular importance cache file: " + str(path))
        require(root in path.resolve().parents, "Importance cache path escapes its root")
        info = record(path)
        sources.append(info)
        return read(path), info

    report, report_record = source("ratings_report.json")
    require(report_sha256 and report_record["sha256"] == report_sha256, "Original importance report checksum mismatch")
    original_manifest, _ = source("original_manifest.json")
    original_manual, _ = source("patrol_manual.json")
    original_classes, _ = source("rating_classes.json")
    current_manual, current_classes = manual_rules(manual), class_names(classes)
    require(report.get("version") == VERSION, "Cached importance protocol mismatch")
    require(report.get("model") == model == MODEL, "Cached requested-model identity mismatch")
    require(report.get("temperature") == 0 and report.get("repeats") == 5, "Cached sampling protocol mismatch")
    require(report.get("status") == "UNVERIFIED", "Original semantic-validation status changed")
    require(current_manual == report["manual"] == manual_rules(original_manual), "Current patrol manual differs from cached model requests")
    require(current_classes == report["class_names"] == class_names(original_classes) == sorted(report["classes"]), "Current object-class list differs from cached model requests")
    require(report.get("score_mapping") == "mean(rating)/5", "Cached score normalization mismatch")
    for field in ("version", "model", "temperature", "repeats", "manual", "class_names", "prompt_sha256", "score_mapping"):
        require(original_manifest[field] == report[field], "Original cache manifest/report mismatch: " + field)

    reconstructed = {}
    for name in current_classes:
        original = report["classes"][name]
        key = original["request_key"]
        require(re.fullmatch(r"[0-9a-f]{64}", key) is not None, "Invalid class request key")
        request, _ = source("classes/" + key + "/request.json")
        identity = {k: v for k, v in request.items() if k != "cache_key"}
        require(identity_hash(identity) == request["cache_key"] == key, "Cached class request identity hash mismatch: " + name)
        require(request["class_name"] == name, "Cached class request name mismatch")
        for field in ("version", "model", "temperature", "manual"):
            require(request[field] == report[field], "Class request/report mismatch: " + field)
        trials = [source("classes/" + key + f"/repeat_{repeat:02d}.json")[0] for repeat in range(1, 6)]
        require(trials == original["trials"], "Saved response files differ from original report: " + name)
        ratings = []
        for repeat, trial in enumerate(trials, 1):
            require(trial["repeat"] == repeat and trial["request_key"] == key, "Saved repeat request identity mismatch")
            response = trial["response"]
            require(set(response) == {"rating", "reason"}, "Invalid saved model response fields")
            require(type(response["rating"]) is int and 1 <= response["rating"] <= 5, "Saved rating is not an integer from 1 to 5")
            require(isinstance(response["reason"], str) and bool(response["reason"].strip()), "Missing saved model reason")
            ratings.append(response["rating"])
        summary = dict(class_name=name, ratings=ratings, n=5, mean=statistics.mean(ratings), sd=statistics.stdev(ratings),
                       minimum=min(ratings), maximum=max(ratings), range=max(ratings)-min(ratings), unanimous=len(set(ratings)) == 1,
                       score=statistics.mean(ratings)/5, request_key=key, trials=trials)
        require(summary == original, "Reconstructed class summary differs from original report: " + name)
        reconstructed[name] = {k: summary[k] for k in ("class_name", "ratings", "n", "mean", "sd", "score", "request_key")}
    return dict(version=VERSION, model=MODEL, temperature=0, repeats=5, manual=current_manual, class_names=current_classes,
                classes=reconstructed, status="UNVERIFIED", score_mapping="mean(rating)/5", manual_hash=identity_hash(current_manual),
                class_list_hash=identity_hash(current_classes), source_report=report_record, source_files=sources,
                mode="cached_response_reconstruction", current_model_calls=0, provider_returned_model_version=None)


def current_scene_scores(scene, verified):
    """Bind reconstructed class scores to this scene's IDs, never old IDs."""
    scores = {}
    for item in scene["objects"]:
        oid, name = item["instance_id"], item.get("class_name")
        require(isinstance(oid, str) and oid and oid not in scores, "Missing or duplicate current object ID")
        require(name in verified["classes"], "Current scene object has no verified cached class rating: " + str(name))
        rating = verified["classes"][name]
        scores[oid] = dict(score=rating["score"], class_name=name,
                           reason=f"Cached five-response mean {rating['mean']:.2f}/5; requested model {verified['model']}",
                           ratings=rating["ratings"], class_mean=rating["mean"], sd=rating["sd"], request_key=rating["request_key"])
    return scores


def importance(job):
    options = job["config"].get("paper_integration", {}).get("importance", {})
    require(options.get("cache_root"), "Configure paper_integration.importance.cache_root for offline cache reuse")
    requested = job["config"]["requested_inputs"]
    capture = job["capture"]["id"] if isinstance(job["capture"], dict) else job["capture"]
    scene_path = Path(job["dependencies"][capture + ".scene"]["scene"])
    scene = read(scene_path)
    require(scene["capture_id"] == capture, "Current importance scene/capture mismatch")
    verified = verify_cached_ratings(options["cache_root"], requested["manual"], requested["object_classes"],
                                     options.get("model_requested", MODEL), options.get("source_report_sha256", REPORT_SHA256))
    scores = current_scene_scores(scene, verified)
    out = Path(job["output"])
    out.mkdir(parents=True, exist_ok=True)
    verified_path = out / "verified_ratings.json"
    save(verified_path, verified)
    paper_label = options.get("paper_model_label", "Gemini 3 Flash-Lite")
    provenance = dict(scene=record(scene_path), source_report=verified["source_report"], cache_root=str(Path(options["cache_root"]).resolve()),
                      manual_hash=verified["manual_hash"], class_list_hash=verified["class_list_hash"],
                      model_requested=verified["model"], provider_returned_model_version=None, repeats=5,
                      current_model_calls=0, mode="cached_response_reconstruction_and_current_scene_id_binding")
    cache = dict(scene_id=scene["scene_id"], capture_id=capture, scores=scores, version=VERSION, status="UNVERIFIED",
                 manual_hash=verified["manual_hash"], model_requested=verified["model"], repeats=5, temperature=0,
                 rating_source=record(verified_path), provenance=provenance,
                 paper_model_label=paper_label,
                 model_naming_note="The preserved requests name gemini-3.1-flash-lite; the publication names Gemini 3 Flash-Lite. Provider returned-model version was not retained.")
    cache_path = out / "importance_cache.json"
    save(cache_path, cache)
    return result(out, cache=str(cache_path), importance_cache=str(cache_path), ratings=str(verified_path),
                  source_report=verified["source_report"]["path"], model_requested=verified["model"], repeats=5,
                  manual_hash=verified["manual_hash"], class_list_hash=verified["class_list_hash"], classes=len(verified["classes"]),
                  current_object_count=len(scores), saved_response_count=5*len(verified["classes"]),
                  status="UNVERIFIED", current_model_calls=0, provenance=provenance,
                  external_artifacts=[str(scene_path.resolve())]+[item["path"] for item in verified["source_files"]])
