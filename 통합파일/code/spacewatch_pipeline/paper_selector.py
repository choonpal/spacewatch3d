"""Paper selector on newly measured native V/Q arrays; no stored frame choices.

The native planner continues to generate observations and report all original
targets/holdouts. Only its selection decision is replaced. No NN is imported.
"""
from __future__ import annotations

from collections import OrderedDict
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import time

for _key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_key, "1")
import numpy as np

DEFAULTS = {"max_views": 15, "deadline_seconds": 35.0}
REFERENCE_SOURCES = {
    "total_object_stop.py": "18ad734bdc6c22a0f9c70d9ff36ffc8f596522c5827f71cdbd1e7c0a3bc20a28",
    "attempt02/benchmark.py": "9df8f4a70338e34adf01734f0d5ad121e5d514eaab17505eabddb05034fb1477",
}


def _save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".partial")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    temporary.replace(path)


def array_record(value):
    a = np.asarray(value)
    h = hashlib.sha256()
    h.update(a.dtype.str.encode())
    h.update(json.dumps(a.shape).encode())
    if a.flags.c_contiguous:
        h.update(memoryview(a).cast("B"))
    else:
        h.update(a.tobytes(order="C"))
    return {"shape": list(a.shape), "dtype": a.dtype.str, "sha256": h.hexdigest()}


def spatial_coefficients(visible, weights, regions, declared=()):
    """Original equal-region, observable-area-normalized spatial term."""
    observable = visible.any(axis=0)
    masks = [regions == rid for rid in sorted(set(regions.tolist()) | set(declared))]
    masks = [m for m in masks if weights[m & observable].sum() > 0]
    coef = np.zeros(len(weights), dtype=np.float64)
    for mask in masks:
        coef[mask] = weights[mask] / weights[mask & observable].sum() / len(masks)
    return coef


def below_stop(gain, threshold):
    # Original strict comparison: equality remains eligible.
    return gain < threshold - 1e-12


class CPUBackend:
    def __init__(self, draws):
        if len(draws) != 3:
            raise ValueError("Paper selection requires exactly three freshly measured optimization draws")
        self.draws = []
        self.n = None
        for raw in draws:
            v = np.asarray(raw["visible"], dtype=bool)
            q = np.asarray(raw["quality"], dtype=np.float64)
            w = np.asarray(raw["weights"], dtype=np.float64)
            r = np.asarray(raw["regions"])
            importance = np.asarray(raw["importance"], dtype=np.float64)
            if v.ndim != 2 or q.ndim != 2 or v.shape[0] != q.shape[0] or v.shape[1] != len(w) or len(r) != len(w) or q.shape[1] != len(importance):
                raise ValueError("Invalid native V/Q/weight/region/object dimensions")
            if self.n is not None and v.shape[0] != self.n:
                raise ValueError("Optimization draws have different candidate ordering/counts")
            if not (np.isfinite(q).all() and np.isfinite(w).all() and np.isfinite(importance).all()) or (q < 0).any() or (w < 0).any() or (importance < 0).any():
                raise ValueError("Non-finite or negative native observation data")
            self.n = v.shape[0]
            iw = importance / importance.sum() if importance.sum() else np.zeros_like(importance)
            self.draws.append({"visible": v, "quality": q, "weights": w, "regions": r,
                               "space_coef": spatial_coefficients(v, w, r, raw.get("declared_regions", ())),
                               "importance_coef": iw})
        self._cache = OrderedDict()

    def components(self, selected):
        result = []
        for d in self.draws:
            cover = d["visible"][selected].any(axis=0) if selected else np.zeros(len(d["weights"]), bool)
            best = d["quality"][selected].max(axis=0) if selected else np.zeros(len(d["importance_coef"]))
            result.append((float(d["space_coef"] @ cover), float(d["importance_coef"] @ best)))
        return np.asarray(result).mean(axis=0)

    def utility(self, selected, sw, ow):
        s, o = self.components(selected)
        return float(sw*s + ow*o)

    def gains(self, selected, sw, ow):
        key = (tuple(sorted(selected)), sw, ow)
        if key in self._cache:
            self._cache.move_to_end(key)
            return self._cache[key].copy()
        score = np.zeros(self.n, dtype=np.float64)
        for d in self.draws:
            if sw:
                cover = d["visible"][selected].any(axis=0) if selected else np.zeros(len(d["weights"]), bool)
                score += sw * np.einsum("ij,j->i", d["visible"], d["space_coef"] * (~cover), dtype=np.float64, optimize=False) / 3
            if ow and len(d["importance_coef"]):
                best = d["quality"][selected].max(axis=0) if selected else np.zeros(len(d["importance_coef"]))
                score += ow * (np.maximum(d["quality"] - best, 0) @ d["importance_coef"]) / 3
        score[np.asarray(selected, dtype=int)] = -np.inf
        self._cache[key] = score.copy()
        if len(self._cache) > 128:
            self._cache.popitem(last=False)
        return score

    def best_pair(self, base, pool, sw, ow):
        pool = list(pool)
        scores = np.zeros((len(pool), len(pool)), dtype=np.float64)
        for d in self.draws:
            if sw:
                cover = d["visible"][base].any(axis=0) if base else np.zeros(len(d["weights"]), bool)
                vv, ww = d["visible"][pool], d["space_coef"] * (~cover)
                one = vv @ ww
                scores += sw * (one[:, None] + one[None, :] - (vv * ww) @ vv.T) / 3
            if ow and len(d["importance_coef"]):
                best = d["quality"][base].max(axis=0) if base else np.zeros(len(d["importance_coef"]))
                qq = np.maximum(d["quality"][pool] - best, 0)
                scores += ow * (np.maximum(qq[:, None, :], qq[None, :, :]) @ d["importance_coef"]) / 3
        scores[np.tril_indices(len(pool))] = -np.inf
        i, j = np.unravel_index(np.argmax(scores), scores.shape)
        return [int(pool[i]), int(pool[j])], float(scores[i, j])


def object_ceiling(backend):
    return float(np.mean([d["quality"].max(axis=0) @ d["importance_coef"] if backend.n else 0. for d in backend.draws]))


def stopping(backend, selected, sw, ow, space_threshold, ceiling):
    gs = max(0., float(np.max(backend.gains(selected, 1., 0.)))) if backend.n else 0.
    go = max(0., float(np.max(backend.gains(selected, 0., 1.)))) if backend.n else 0.
    sd = not sw or below_stop(gs, space_threshold) or gs <= 1e-12
    od = not ow or ceiling <= 1e-12 or below_stop(go, .03 * ceiling) or go <= 1e-12
    return bool(sd and od), {"max_space_gain": gs, "space_threshold": space_threshold,
                            "max_object_gain": go, "object_attainable_score": ceiling,
                            "object_gain_ratio": go / ceiling if ceiling > 1e-12 else None,
                            "object_threshold": .03 * ceiling, "space_done": bool(sd), "object_done": bool(od)}


def exchange(backend, selected, sw, ow, deadline, required=()):
    """Original six one-swap passes, then a top-12 two-swap search."""
    selected = list(selected)
    history = []
    for _ in range(6):
        if time.monotonic() >= deadline:
            break
        current, proposal = backend.utility(selected, sw, ow), None
        for removed in selected:
            if removed in required:
                continue
            base = [i for i in selected if i != removed]
            gain = backend.gains(base, sw, ow)
            new = int(np.argmax(gain))
            trial = base + [new]
            value = backend.utility(trial, sw, ow)
            if value > current + 1e-10 and (proposal is None or value > proposal[0] + 1e-12):
                proposal = value, trial, [removed], [new]
            if time.monotonic() >= deadline:
                break
        if proposal is None:
            break
        score, selected, removed, added = proposal
        history.append({"kind": "one_swap", "removed": removed, "added": added, "score": score})
    if len(selected) >= 2:
        current, proposal = backend.utility(selected, sw, ow), None
        for removed in itertools.combinations([i for i in selected if i not in required], 2):
            if time.monotonic() >= deadline:
                break
            base = [i for i in selected if i not in removed]
            gains = backend.gains(base, sw, ow)
            pool = [int(i) for i in np.argsort(-gains, kind="stable") if np.isfinite(gains[i])][:12]
            if len(pool) < 2:
                continue
            added, _ = backend.best_pair(base, pool, sw, ow)
            trial = base + added
            value = backend.utility(trial, sw, ow)
            if value > current + 1e-10 and (proposal is None or value > proposal[0] + 1e-12):
                proposal = value, trial, list(removed), added
        if proposal is not None:
            score, selected, removed, added = proposal
            history.append({"kind": "two_swap_top12", "removed": removed, "added": added, "score": score})
    return selected, history


def _options(options):
    options = {} if options is None else dict(options)
    unknown = set(options) - set(DEFAULTS)
    if unknown:
        raise ValueError("Unknown paper selection options: " + str(sorted(unknown)))
    result = {**DEFAULTS, **options}
    cap = result["max_views"]
    if cap is not None and (isinstance(cap, bool) or not isinstance(cap, int) or cap < 1):
        raise ValueError("max_views must be positive integer or null")
    seconds = result["deadline_seconds"]
    if isinstance(seconds, bool) or not isinstance(seconds, (float, int)) or not math.isfinite(seconds) or seconds <= 0:
        raise ValueError("deadline_seconds must be positive finite")
    return result


def select_draws(draws, space_weight, object_weight, required=(), options=None):
    options = _options(options)
    started = time.monotonic()
    deadline = started + options["deadline_seconds"]
    backend = CPUBackend(draws)
    required = list(dict.fromkeys(int(i) for i in required))
    if any(i < 0 or i >= backend.n for i in required):
        raise ValueError("Required candidate index out of range")
    cap = min(backend.n, options["max_views"] if options["max_views"] is not None else backend.n)
    if len(required) > cap:
        raise ValueError("Required candidates exceed configured view ceiling")
    sw, ow = float(space_weight), float(object_weight)
    if not math.isfinite(sw + ow) or sw < 0 or ow < 0 or sw + ow <= 0:
        raise ValueError("Invalid paper objective weights")
    if not any(d["importance_coef"].sum() for d in backend.draws):
        ow = 0.
        if sw == 0:
            sw = 1.
    normalizer = sw + ow
    sw, ow = sw / normalizer, ow / normalizer
    space_threshold = .02 * max(0., float(np.max(backend.gains([], 1., 0.)))) if backend.n else 0.
    ceiling = object_ceiling(backend)
    done, info = stopping(backend, required, sw, ow, space_threshold, ceiling)
    seeds, used = [], required.copy()
    for _ in range(min(6, backend.n - len(required))):
        gains = backend.gains(used, sw, ow)
        seed = int(np.argmax(gains))
        seeds.append(seed)
        used.append(seed)
    frontiers = [required.copy()] + [required + [seed] for seed in seeds]
    curve, chosen, reason = [], required.copy(), "view_cap"
    if done:
        reason = "marginal_conditions_met"
    else:
        for k in range(max(1, len(required)), cap + 1):
            available = []
            for row in frontiers:
                while len(row) < k:
                    gains = backend.gains(row, sw, ow)
                    row.append(int(np.argmax(gains)))
                if len(row) == k:
                    available.append(row.copy())
                if time.monotonic() >= deadline:
                    break
            if not available:
                reason = "time_limit"
                break
            prior = max(available, key=lambda s: backend.utility(s, sw, ow))
            chosen, changes = exchange(backend, prior, sw, ow, deadline, required)
            done, info = stopping(backend, chosen, sw, ow, space_threshold, ceiling)
            curve.append({"k": len(chosen), "selected": chosen.copy(), "score": backend.utility(chosen, sw, ow),
                          "history": changes, "stop_met": done, "stop_details": info})
            if time.monotonic() >= deadline:
                reason = "time_limit"
                break
            if done:
                reason = "marginal_conditions_met"
                break
    score_parts = backend.components(chosen)
    return {"schema": "spacewatch.paper_selector.v1", "selected": chosen, "count": len(chosen),
            "candidate_count": backend.n, "draw_count": 3, "options": options,
            "weights": {"space": sw, "object": ow}, "space_score": float(score_parts[0]),
            "raw_object_score": float(score_parts[1]), "score": backend.utility(chosen, sw, ow),
            "stop_reason": reason, "marginal_stop_verified": reason == "marginal_conditions_met",
            "stop_details": info, "curve": curve, "diverse_seeds": seeds,
            "unseeded_greedy_start": True, "required": required,
            "warm_start_bank_used": False, "holdouts_used_for_selection": False,
            "global_optimum_claimed": False, "elapsed_seconds": time.monotonic() - started,
            "reference_source_sha256": REFERENCE_SOURCES,
            "policy": {"space_stop": "max remaining spatial gain < 0.02 * max empty-set singleton spatial gain",
                       "object_stop": "max remaining raw object gain < 0.03 * full-pool attainable raw object score",
                       "mixed_stop": "both active terms stopped, after local exchanges",
                       "utility": "mean over three draws of alpha*equal-region-space + (1-alpha)*importance-weighted max raw q",
                       "exchange": "up to six one-swap passes then one top12 two-swap search per cardinality"}}


def install(planner, receipt_path, options=None):
    """Install into a native planner process; return an idempotent restore hook."""
    options = _options(options)
    original_make, original_select, original_create = planner.make_objectives, planner.select_keyframes, planner.create_plan
    state = {}

    def make_objectives(datasets, allowed_indices, objects, config, declared_regions):
        result = original_make(datasets, allowed_indices, objects, config, declared_regions)
        state["draw_seeds"] = [d["seed"] for d in datasets]
        state["allowed_indices"] = list(allowed_indices)
        state["objects"] = [o["instance_id"] for o in objects]
        return result

    def select_keyframes(optimization, evaluator=None, required=()):
        objectives = getattr(optimization, "objectives", [optimization])
        config = objectives[0].config
        if evaluator is not None or config.route_aware_selection or config.lateral_candidate_refinement or config.max_link_m is not None:
            raise ValueError("Paper selector requires route_aware_selection=false, lateral_candidate_refinement=false, max_link_m=null")
        if not config.coverage_repair or config.coverage_repair_runs != 2 or len(objectives) != 3:
            raise ValueError("Paper selector requires three measured native draws: coverage_repair=true/runs=2")
        draws = [{"visible": o.visible, "quality": o.quality, "weights": o.weights, "regions": o.regions,
                  "importance": o.importance, "declared_regions": o.declared_regions} for o in objectives]
        result = select_draws(draws, config.space_weight, config.object_weight, required, options)
        result["observations"] = [{"seed": seed, **{k: array_record(d[k]) for k in ("visible", "quality", "weights", "regions", "importance")}}
                                  for seed, d in zip(state["draw_seeds"], draws)]
        result["allowed_candidate_indices"] = state["allowed_indices"]
        result["object_ids"] = state["objects"]
        result["producer_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        state["result"] = result
        _save(receipt_path, result)
        chosen = result["selected"]
        history = {}
        prefix = []
        for step, index in enumerate(chosen, 1):
            before = float(optimization.measure(prefix)[0])
            prefix.append(index)
            history[index] = {"selection_step": step,
                              "marginal_gain_at_selection": float(optimization.measure(prefix)[0]) - before,
                              "marginal_image_count_at_selection": 1,
                              "selection_method": "paper_multistart_exchange"}
        report = optimization.measure(chosen)[1]
        report["paper_selection"] = result
        return chosen, history, report

    def create_plan(*args, **kwargs):
        plan, requests = original_create(*args, **kwargs)
        result = state.get("result")
        if result is None:
            raise RuntimeError("Native planner bypassed paper selector")
        candidates = plan["candidate_report"]["candidates"]
        expected = [candidates[state["allowed_indices"][i]]["route_node_id"] for i in result["selected"]]
        actual = [v["route_node_id"] for v in plan["viewpoints"]]
        if len(actual) != len(expected) or set(actual) != set(expected):
            result["postselection_identity_verified"] = False
            result["expected_route_node_ids"], result["actual_route_node_ids"] = expected, actual
            _save(receipt_path, result)
            raise RuntimeError("Native postselection changed paper-selected frame set")
        result["postselection_identity_verified"] = True
        result["selected_route_node_ids"] = expected
        plan["coverage_report"]["paper_selection"] = result
        plan["coverage_report"]["selection"]["strategy"] = "paper_multistart_one_two_exchange_separate_marginal_stop"
        plan["provenance"]["paper_selection"] = {"receipt_path": str(receipt_path), "producer_sha256": result["producer_sha256"],
                                                       "fresh_native_observations": True, "holdouts_used_for_selection": False}
        if not result["marginal_stop_verified"]:
            plan["planning_status"] = "incomplete"
            plan["failures"].append({"code": "paper_selection_stopping_unverified", "stop_reason": result["stop_reason"]})
        _save(receipt_path, result)
        return plan, requests

    planner.make_objectives, planner.select_keyframes, planner.create_plan = make_objectives, select_keyframes, create_plan

    def restore():
        planner.make_objectives, planner.select_keyframes, planner.create_plan = original_make, original_select, original_create
    return restore
