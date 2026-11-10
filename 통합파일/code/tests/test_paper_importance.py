import hashlib
import json
from pathlib import Path
import statistics
import tempfile
import unittest

from spacewatch_pipeline.paper_importance import (
    MODEL, VERSION, current_scene_scores, identity_hash, importance, manual_rules, verify_cached_ratings,
)


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def fixture(root):
    manual = ["Check doors.", "Check chairs."]
    ratings = {"door": [5, 5, 5, 5, 5], "chair": [2, 3, 3, 3, 4]}
    base = dict(version=VERSION, model=MODEL, temperature=0, repeats=5, manual=manual,
                class_names=sorted(ratings), prompt_sha256="fixture-prompt", score_mapping="mean(rating)/5")
    classes = {}
    for name, scores in ratings.items():
        request = dict(version=VERSION, model=MODEL, temperature=0, manual=manual, class_name=name,
                       prompt="Fixture only: " + name, response_schema={})
        key = identity_hash(request)
        request["cache_key"] = key
        directory = root / "classes" / key
        write(directory / "request.json", request)
        trials = []
        for repeat, rating in enumerate(scores, 1):
            trial = dict(repeat=repeat, request_key=key, response=dict(rating=rating, reason="Unit fixture response"), usage={}, seconds=0)
            trials.append(trial)
            write(directory / f"repeat_{repeat:02d}.json", trial)
        classes[name] = dict(class_name=name, ratings=scores, n=5, mean=statistics.mean(scores), sd=statistics.stdev(scores),
                             minimum=min(scores), maximum=max(scores), range=max(scores)-min(scores), unanimous=len(set(scores)) == 1,
                             score=statistics.mean(scores)/5, request_key=key, trials=trials)
    write(root / "original_manifest.json", base)
    write(root / "ratings_report.json", {**base, "classes": classes, "status": "UNVERIFIED"})
    write(root / "patrol_manual.json", {"M01": manual[0], "M02": manual[1]})
    write(root / "rating_classes.json", list(ratings))
    pin = hashlib.sha256((root / "ratings_report.json").read_bytes()).hexdigest()
    return manual, list(ratings), pin


class CachedCurrentImportance(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix=".importance-test-", dir=Path(__file__).resolve().parents[1])
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.cache = self.root / "cache"
        self.manual, self.names, self.pin = fixture(self.cache)

    def verified(self, **kwargs):
        return verify_cached_ratings(self.cache, self.manual, self.names, report_sha256=self.pin, **kwargs)

    def test_reconstructs_all_five_responses_and_normalizes_mean(self):
        data = self.verified()
        self.assertEqual(data["classes"]["door"]["score"], 1)
        self.assertEqual(data["classes"]["chair"]["score"], .6)
        self.assertEqual(data["classes"]["chair"]["ratings"], [2, 3, 3, 3, 4])
        self.assertEqual(data["current_model_calls"], 0)
        self.assertIsNone(data["provider_returned_model_version"])

    def test_binds_current_ids_and_reuses_class_score_for_each_object(self):
        scene = {"objects": [{"instance_id": "new-door-A", "class_name": "door"}, {"instance_id": "new-door-B", "class_name": "door"}, {"instance_id": "new-chair", "class_name": "chair"}]}
        scores = current_scene_scores(scene, self.verified())
        self.assertEqual(set(scores), {"new-door-A", "new-door-B", "new-chair"})
        self.assertEqual(scores["new-door-A"]["score"], scores["new-door-B"]["score"])

    def test_manual_and_class_identity_mismatches_rejected(self):
        with self.assertRaisesRegex(ValueError, "patrol manual differs"):
            verify_cached_ratings(self.cache, ["Different rule."], self.names, report_sha256=self.pin)
        with self.assertRaisesRegex(ValueError, "class list differs"):
            verify_cached_ratings(self.cache, self.manual, ["door"], report_sha256=self.pin)
        with self.assertRaisesRegex(ValueError, "Duplicate current"):
            verify_cached_ratings(self.cache, self.manual, ["door", "door"], report_sha256=self.pin)

    def test_request_model_and_report_pin_mismatches_rejected(self):
        with self.assertRaisesRegex(ValueError, "requested-model"):
            self.verified(model="gemini-2.5-flash-lite")
        with self.assertRaisesRegex(ValueError, "checksum"):
            verify_cached_ratings(self.cache, self.manual, self.names, report_sha256="0"*64)

    def test_tampered_response_and_request_rejected(self):
        key = self.verified()["classes"]["door"]["request_key"]
        path = self.cache / "classes" / key / "repeat_01.json"
        original = json.loads(path.read_text())
        modified = {**original, "response": {**original["response"], "rating": 1}}
        write(path, modified)
        with self.assertRaisesRegex(ValueError, "Saved response files differ"):
            self.verified()
        write(path, original)
        path = self.cache / "classes" / key / "request.json"
        modified = json.loads(path.read_text())
        modified["class_name"] = "chair"
        write(path, modified)
        with self.assertRaisesRegex(ValueError, "identity hash"):
            self.verified()

    def test_missing_repeat_and_unknown_current_class_fail_without_defaults(self):
        verified = self.verified()
        with self.assertRaisesRegex(ValueError, "no verified cached class"):
            current_scene_scores({"objects": [{"instance_id": "new", "class_name": "unknown"}]}, verified)
        key = verified["classes"]["door"]["request_key"]
        (self.cache / "classes" / key / "repeat_05.json").unlink()
        with self.assertRaisesRegex(ValueError, "Missing regular"):
            self.verified()

    def test_stage_produces_native_cache_and_provenance(self):
        scene = self.root / "scene.json"
        write(scene, {"scene_id": "current-scene", "capture_id": "after", "objects": [{"instance_id": "current-object", "class_name": "chair"}]})
        job = dict(capture={"id": "after"}, output=str(self.root / "output"), dependencies={"after.scene": {"scene": str(scene)}},
                   config={"requested_inputs": {"manual": self.manual, "object_classes": self.names},
                           "paper_integration": {"importance": {"cache_root": str(self.cache), "source_report_sha256": self.pin}}})
        result = importance(job)
        cache = json.loads(Path(result["cache"]).read_text())
        self.assertEqual(result["cache"], result["importance_cache"])
        self.assertEqual(cache["capture_id"], "after")
        self.assertEqual(cache["scores"]["current-object"]["score"], .6)
        self.assertEqual(cache["model_requested"], MODEL)
        self.assertEqual(cache["rating_source"]["path"], result["ratings"])
        self.assertEqual(result["saved_response_count"], 10)
        self.assertEqual(result["current_model_calls"], 0)
        self.assertNotEqual(cache["paper_model_label"], cache["model_requested"])

    def test_preprocessing_matches_saved_rule_normalization(self):
        self.assertEqual(manual_rules("M01: Check  doors.\n2. Check chairs."), self.manual)
        self.assertEqual(manual_rules({"M01": "Check doors.", "M02": "Check chairs."}), self.manual)


if __name__ == "__main__":
    unittest.main()
