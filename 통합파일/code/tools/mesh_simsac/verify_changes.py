"""Object-level verification of change candidates (runs after build_event_viewer).

One physical change is often reported as several neighbouring candidates (a coat on a chair shows up on the
chair and on the table), and furniture next to a person changes appearance with the person. Candidates are
grouped in the common map frame using measured sizes only; no distance threshold is tuned:
  * same change     distance <= r1 + r2 + e, complete linkage (every pair in a group must satisfy it,
                    so a large object cannot chain two separate changes together)
  * person context  a non-person candidate within r_person + r + e joins that person's change
  * both sessions   RGB candidates of the same label seen in opposite sessions within max(r1, r2) + e cancel
  * windows         window candidates are excluded (reflections)
r: half the bounds diagonal (mesh) or half the 10-90 percentile box diagonal of the RGB mask points.
e: median registration error of the mesh-backed candidates of this run.
"""
import itertools
import json
from pathlib import Path

import numpy as np


def read(path): return json.loads(Path(path).read_text())
def write(path, value): Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def extent(event):
    if event.get("bounds"):
        bounds = np.asarray(event["bounds"], float)
        return bounds.mean(0), float(np.linalg.norm(bounds[1] - bounds[0]) / 2)
    if event.get("center") is not None:
        return np.asarray(event["center"], float), float(event["radius_u"])
    return None, None


def verify(root):
    root = Path(root); data = read(root / "viewer-data.json"); events = data["events"]
    errors = [e["registration_error_u"] for e in events if e.get("bounds") and e.get("registration_error_u") is not None]
    error = float(np.median(errors)) if errors else float(data["policy"]["registration_error_u"])
    geometry = {e["id"]: extent(e) for e in events}; label = {e["id"]: e["label"] for e in events}
    dist = lambda a, b: float(np.linalg.norm(geometry[a][0] - geometry[b][0]))
    radius = lambda a: geometry[a][1]
    status = {}
    for e in events:
        if e["label"] == "window": status[e["id"]] = "excluded_window_reflection"
        elif geometry[e["id"]][0] is None: status[e["id"]] = "kept_without_location"
    rgb = [e for e in events if e.get("rgb_only") and e["id"] not in status]
    for a, b in itertools.combinations(rgb, 2):
        if a["label"] == b["label"] and bool(a["before"]) != bool(b["before"]) \
                and dist(a["id"], b["id"]) <= max(radius(a["id"]), radius(b["id"])) + error:
            status[a["id"]] = status[b["id"]] = "cancelled_seen_in_both_sessions"
    live = [e["id"] for e in events if e["id"] not in status]
    persons = [i for i in live if label[i] == "person"]; groups = {p: [p] for p in persons}; rest = []
    for i in live:
        if i in persons: continue
        near = [(dist(i, p), p) for p in persons if dist(i, p) <= radius(i) + radius(p) + error]
        if near: groups[min(near)[1]].append(i)
        else: rest.append(i)
    items = list(groups.values()) + [[i] for i in rest]
    linked = lambda A, B: all(dist(a, b) <= radius(a) + radius(b) + error for a in A for b in B)
    while True:
        pairs = [(max(dist(a, b) for a in items[x] for b in items[y]), x, y)
                 for x, y in itertools.combinations(range(len(items)), 2) if linked(items[x], items[y])]
        if not pairs: break
        _, x, y = min(pairs); items[x] = items[x] + items[y]; del items[y]
    items += [[i] for i, s in status.items() if s == "kept_without_location"]
    changes = []
    for members in sorted(items, key=min):
        located = [m for m in members if geometry[m][0] is not None]
        lead = next((m for m in members if label[m] == "person"), min(located, key=radius) if located else members[0])
        changes.append(dict(id=f"CH-{len(changes)+1:02d}", label=label[lead], lead_event=lead, event_ids=sorted(members),
                            center=np.mean([geometry[m][0] for m in located], 0).tolist() if located else None))
        for m in members:
            status.setdefault(m, "person_context" if label[lead] == "person" and m != lead else "grouped")
    owner = {m: c["id"] for c in changes for m in c["event_ids"]}
    for e in events:
        e["verified_change"] = owner.get(e["id"]); e["verification"] = status[e["id"]]
        if geometry[e["id"]][0] is not None:
            e["verification_extent"] = dict(center=geometry[e["id"]][0].tolist(), radius_u=geometry[e["id"]][1])
    policy = dict(rules=["same_change: distance <= r1 + r2 + e, complete linkage",
                         "person_context: distance <= r_person + r + e",
                         "both_sessions: same RGB label, opposite sessions, distance <= max(r1, r2) + e",
                         "window: excluded"],
                  registration_error_u=error, radius="mesh: half bounds diagonal; RGB: half 10-90 percentile box diagonal",
                  tuned_thresholds=False, ground_truth_used=False)
    write(root / "results/verified-changes.json", dict(policy=policy, changes=changes,
          events={e["id"]: dict(verified_change=e["verified_change"], verification=e["verification"]) for e in events}))
    data["verified_changes"] = changes; data["verification_policy"] = policy
    data["counts"] = {**data["counts"], "verified_changes": len(changes)}
    write(root / "viewer-data.json", data)
    return changes
