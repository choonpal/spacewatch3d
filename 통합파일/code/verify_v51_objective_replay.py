#!/usr/bin/env python3
"""CPU evaluator replay from this project's locally captured trusted pickle."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import pickle
import statistics
import sys
import time
import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path[:0] = [str(ROOT/'vendor/spacewatch3d_src'), str(ROOT/'vendor/native')]
import objective_executor as current


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--replay', required=True, type=Path)
    parser.add_argument('--baseline-executor', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    spec = importlib.util.spec_from_file_location('baseline_objective_executor', args.baseline_executor)
    baseline = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(baseline)
    data = args.replay.read_bytes()
    first = pickle.loads(data)['optimization']
    n = len(first.visible)
    rng = np.random.default_rng(20261026)
    queries = [[], list(range(n))] + [rng.choice(n, k, replace=False).tolist()
               for k in (1, 13, 50, 100) for _ in range(100)]
    modes = {'baseline': baseline, 'deduplicated_sums': current}
    measurements = {name: [] for name in modes}
    reference = None
    for repeat in range(5):
        names = list(modes) if repeat % 2 == 0 else list(reversed(modes))
        for name in names:
            modes[name].install(immutable=True)
            objective = pickle.loads(data)['optimization']
            start = time.perf_counter()
            reports = [objective.measure(q) for q in queries]
            seconds = time.perf_counter()-start
            measurements[name].append(seconds)
            if reference is None:
                reference = reports
            assert reference == reports, 'Evaluator reports changed'
            print(name, repeat, seconds, flush=True)
    current.restore()
    medians = {name: statistics.median(values) for name, values in measurements.items()}
    result = dict(scope='local CPU real captured objectives; not full planner or server time',
                  queries=len(queries), repeats=5, exact_full_report_parity=True,
                  visibility_shapes=[list(o.visible.shape) for o in first.objectives],
                  replay_sha256=hashlib.sha256(data).hexdigest(),
                  baseline_executor_sha256=hashlib.sha256(args.baseline_executor.read_bytes()).hexdigest(),
                  executor_sha256=hashlib.sha256(Path(current.__file__).read_bytes()).hexdigest(),
                  seconds=measurements, median_seconds=medians,
                  median_reduction_percent=100*(1-medians['deduplicated_sums']/medians['baseline']))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
