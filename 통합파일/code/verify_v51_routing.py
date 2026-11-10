#!/usr/bin/env python3
"""CPU-only replay of saved real route graphs, not a full-pipeline benchmark."""
import argparse
from dataclasses import replace
from functools import lru_cache
import hashlib
import importlib.util
import json
from pathlib import Path
import statistics
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'vendor/spacewatch3d_src'))
from spacewatch3d.tasks.task_5b_keyframes.models import Config
from spacewatch3d.tasks.task_5b_keyframes.routing import Route, connect_tour, shortest


def without_geometry(result):
    visits, links, transitions, failures = result
    return visits, [{k: v for k, v in link.items()
                     if k not in ('route_node_ids', 'polyline_m')} for link in links], transitions, failures


def load_route(plan):
    nodes = plan['route_graph']['nodes']
    lookup = {node['node_id']: i for i, node in enumerate(nodes)}
    adjacency = {i: [] for i in range(len(nodes))}
    for edge in plan['route_graph']['edges']:
        a, b = lookup[edge['from_node']], lookup[edge['to_node']]
        adjacency[a].append((b, edge['length_m']))
        adjacency[b].append((a, edge['length_m']))
    route = Route(origin=np.zeros(2), grid_m=plan['route_graph']['grid_m'],
                  walkable=np.empty((0, 0)), clearance=np.empty((0, 0)),
                  cells=[(0, 0)] * len(nodes),
                  positions=np.array([n['position_m'] for n in nodes]),
                  components=[int(n['component_id'].split('_')[-1]) for n in nodes],
                  adjacency=adjacency,
                  candidates=[lookup[c['route_node_id']] for c in plan['candidate_report']['candidates']],
                  obstacles=None, node_ids={i: n['node_id'] for i, n in enumerate(nodes)})
    return route, Config.from_dict(plan['provenance']['config'])


def benchmark(path, baseline, queries=150, repeats=5):
    plan = json.loads(path.read_text())
    route, config = load_route(plan)
    rng = np.random.default_rng(20261026)
    selected = [rng.choice(route.candidates, min(13, len(route.candidates)), replace=False).tolist()
                for _ in range(queries)]

    def runner(function, include_geometry, config=config, banned=frozenset()):
        paths = lru_cache(maxsize=32)(lambda n: shortest(route.adjacency, n))
        hops = lru_cache(maxsize=256)(lambda n: shortest(route.adjacency, n,
            max_distance=config.max_link_m if config.max_link_m is not None else float('inf')))
        kwargs = {} if include_geometry else {'include_geometry': False}
        return lambda nodes: function(route, nodes, config, banned,
                                      paths_from=paths, hop_paths_from=hops, **kwargs)

    # Check full default exports against the pre-change function, then score-only
    # fields exactly, including capped, uncapped and rejected-image-stop paths.
    checked = 0
    for limit, banned in [(config.max_link_m, frozenset()), (None, frozenset()),
                          (.15, frozenset(route.candidates[::3]))]:
        settings = replace(config, max_link_m=limit)
        old = runner(baseline, True, settings, banned)
        full = runner(connect_tour, True, settings, banned)
        fast = runner(connect_tour, False, settings, banned)
        cases = [[], [route.candidates[0]]] + selected
        for nodes in cases:
            nodes = [n for n in nodes if n not in banned]
            expected = old(nodes)
            assert expected == full(nodes), 'default final geometry changed'
            assert without_geometry(expected) == fast(nodes), 'candidate route scores changed'
            checked += 1
    times = {'baseline': [], 'score_only': []}
    for repeat in range(repeats):
        order = ['baseline', 'score_only'] if repeat % 2 == 0 else ['score_only', 'baseline']
        for name in order:
            run = runner(baseline if name == 'baseline' else connect_tour, name == 'baseline')
            start = time.perf_counter()
            for nodes in selected:
                run(nodes)
            times[name].append(time.perf_counter() - start)
    medians = {name: statistics.median(values) for name, values in times.items()}
    return {'plan': str(path.resolve()), 'plan_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
            'graph_nodes': len(route.positions), 'candidates': len(route.candidates),
            'queries_per_repeat': queries, 'exact_parity_cases': checked,
            'full_default_export_exact': True, 'candidate_score_exact': True,
            'seconds': times, 'median_seconds': medians,
            'median_reduction_percent': 100 * (1 - medians['score_only'] / medians['baseline'])}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', action='append', required=True, type=Path)
    parser.add_argument('--baseline-routing', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--queries', type=int, default=150)
    parser.add_argument('--repeats', type=int, default=5)
    args = parser.parse_args()
    if args.queries < 1 or args.repeats < 1:
        parser.error('queries and repeats must be positive')
    name = 'spacewatch3d.tasks.task_5b_keyframes._routing_baseline'
    spec = importlib.util.spec_from_file_location(name, args.baseline_routing)
    baseline = importlib.util.module_from_spec(spec)
    sys.modules[name] = baseline
    spec.loader.exec_module(baseline)
    report = {'scope': 'local CPU saved-route replay; not full planner or server wall time',
              'baseline_routing_sha256': hashlib.sha256(args.baseline_routing.read_bytes()).hexdigest(),
              'routing_sha256': hashlib.sha256((ROOT / 'vendor/spacewatch3d_src/spacewatch3d/tasks/task_5b_keyframes/routing.py').read_bytes()).hexdigest(),
              'scenes': {path.parent.name: benchmark(path, baseline.connect_tour, args.queries, args.repeats)
                         for path in args.plan}}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
