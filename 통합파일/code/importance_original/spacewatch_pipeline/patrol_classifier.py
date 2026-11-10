"""Isolated role-classification pilot. No transport or production edits."""
from pathlib import Path
import hashlib
import json
import re

VERSION = "patrol-importance-simple-v2"
MODEL = "gemini-3.1-flash-lite"
ROLES = ("target", "location_reference", "generic_only", "exclusion", "not_mentioned")
TEXT_FIELDS = ("line", "quote", "checked_target", "named_term")
FIELDS = set(TEXT_FIELDS) | {"kind_role", "general_relevance", "reason"}
EXCLUDED = {"wall", "walls", "ceiling", "ceilings", "벽", "천장", "ceiling hallway"}
SCHEMA = {"type": "object", "additionalProperties": False, "required": sorted(FIELDS),
          "properties": {**{key: {"anyOf": [{"type": "string"}, {"type": "null"}]} for key in TEXT_FIELDS},
                         "kind_role": {"type": "string", "enum": list(ROLES)},
                         "general_relevance": {"type": "boolean"}, "reason": {"type": "string"}}}


def normalize(text):
    return " ".join(text.split())


def preprocess_manual(text):
    """Input must contain patrol rules only; no automatic semantic deletion."""
    sentences = []
    for line in text.splitlines():
        line = re.sub(r"^\s*(?:[-*•]\s+|\d+[.)]\s+|M\d+\s*[:：]\s*)", "", line)
        line = re.sub(r"^\s*(?:장소별 운영 규칙|순찰 매뉴얼|순찰 기준)(?:\s*\([^)]*\))?\s*[:：]\s*", "", line)
        sentences.extend(normalize(part) for part in re.split(r"(?<=[.!?])\s+", line) if part.strip())
    return {f"M{i:02d}": sentence for i, sentence in enumerate(sentences, 1)}


def eligible(name):
    return name is None or name.strip().casefold() not in EXCLUDED


def prepare_request(name, manual, template):
    if not isinstance(name, str) or not name.strip() or not eligible(name):
        raise ValueError("nonempty_eligible_class_required")
    if not manual or any(not re.fullmatch(r"M\d+", key) or not normalize(value) for key, value in manual.items()):
        raise ValueError("numbered_nonempty_manual_required")
    manual = {key: normalize(value) for key, value in manual.items()}
    prompt = template.replace("{{manual}}", "\n".join(f"{key}: {value}" for key, value in manual.items()))
    prompt = prompt.replace("{{class_name_json}}", json.dumps(name, ensure_ascii=False))
    version = VERSION + ":" + hashlib.sha256((template + json.dumps(SCHEMA, sort_keys=True)).encode()).hexdigest()
    identity = {"prompt_version": version, "model": MODEL, "manual": manual, "class_name": name}
    encoded = json.dumps(identity, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return {**identity, "temperature": 0, "prompt": prompt, "response_schema": SCHEMA,
            "cache_key": hashlib.sha256(encoded.encode()).hexdigest()}


def schema_ok(raw):
    return (isinstance(raw, dict) and set(raw) == FIELDS and raw["kind_role"] in ROLES
            and type(raw["general_relevance"]) is bool
            and isinstance(raw["reason"], str) and bool(raw["reason"].strip())
            and all(raw[key] is None or isinstance(raw[key], str) for key in TEXT_FIELDS))


def validate_response(raw, manual):
    if not schema_ok(raw):
        raise ValueError("invalid_response_schema")
    events = []
    target = raw["kind_role"] == "target"
    if target:
        line = raw["line"]
        if not isinstance(line, str) or line not in manual or not re.fullmatch(r"M\d+", line):
            events.append("invalid_manual_line")
        quote, checked, term = (normalize(raw[key]) if isinstance(raw[key], str) else ""
                                for key in ("quote", "checked_target", "named_term"))
        if not all((quote, checked, term)):
            events.append("empty_target_evidence")
        elif (not isinstance(line, str) or line not in manual or quote not in normalize(manual[line])
              or checked not in quote or term not in checked):
            events.append("invalid_target_quote_nesting")
    level = 2 if target and not events else 1 if target or raw["general_relevance"] else 0
    return {**raw, "level": level, "score": level / 2, "raw_response": raw,
            "policy_events": [{"code": code, "action": "set_level_1", "raw_response": raw} for code in events],
            "verification_status": "UNVERIFIED"}


def classify(request, complete_json, on_attempt=None):
    attempts = []
    for attempt in (1, 2):
        raw = complete_json(request)
        attempts.append(raw)
        if on_attempt:
            on_attempt(attempt, raw)
        if schema_ok(raw):
            result = validate_response(raw, request["manual"])
            break
    else:
        result = {"level": 1, "score": .5, "kind_role": None, "reason": "응답 형식 실패에 따른 기본 중요도",
                  "policy_events": [{"code": "schema_failed_twice", "action": "set_level_1", "raw_responses": attempts}],
                  "verification_status": "UNVERIFIED"}
    result.update(cache_key=request["cache_key"], class_name=request["class_name"], attempts=attempts)
    if len(attempts) > 1:
        result["policy_events"].insert(0, {"code": "schema_retry_once", "raw_response": attempts[0]})
    return result


def write_new(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf8") as stream:
        json.dump(data, stream, ensure_ascii=False, indent=2, allow_nan=False)


def classify_cached(request, directory, complete_json=None):
    directory = Path(directory) / request["cache_key"]
    result_path = directory / "result.json"
    if result_path.exists():
        saved = json.loads(result_path.read_text())
        if saved.get("cache_key") != request["cache_key"] or saved.get("class_name") != request["class_name"]:
            raise ValueError("cache_identity_mismatch")
        values = iter(saved["attempts"])
        actual = classify(request, lambda _: next(values))
        if len(actual["attempts"]) != len(saved["attempts"]):
            raise ValueError("cached_attempt_count_mismatch")
        return actual
    if complete_json is None:
        raise RuntimeError("no_authorized_transport")
    directory.mkdir(parents=True, exist_ok=False)
    write_new(directory / "request.json", request)
    result = classify(request, complete_json,
                      lambda attempt, raw: write_new(directory / f"attempt_{attempt:02d}.json", raw))
    write_new(result_path, result)
    return result


def export_cache(scene, class_results):
    scores = {}
    for item in scene["objects"]:
        name, oid = item.get("class_name"), item["instance_id"]
        if not eligible(name):
            continue
        if oid in scores:
            raise ValueError("duplicate_instance_id")
        result = class_results[name] if name is not None else {"level": 1, "reason": "종류 미분류"}
        level = result["level"]
        if type(level) is not int or level not in (0, 1, 2):
            raise ValueError("invalid_class_level")
        basis = " · ".join(str(result[key]) for key in ("line", "named_term") if result.get(key))
        scores[oid] = {"score": level / 2, "reason": f"L{level}: {basis or result['reason']}"}
    return {"scene_id": scene["scene_id"], "capture_id": scene["capture_id"], "scores": scores}
