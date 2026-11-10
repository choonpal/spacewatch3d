#!/usr/bin/env python3
"""Compare CPU surface deduplication on actual input meshes; no inference."""
import argparse
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import statistics
import sys
import time

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT/'vendor/spacewatch3d_src'))
from spacewatch3d.tasks.task_5b_keyframes import loader
from spacewatch3d.tasks.task_5b_keyframes.models import Config


class Captured(Exception):
    pass


def raw_geometry(scene, config):
    captured = {}
    original = loader.deduplicate_surfaces
    def capture(geometries, _config, _warnings):
        captured['geometry'] = geometries
        raise Captured()
    loader.deduplicate_surfaces = capture
    try:
        loader.load_scene(scene, config)
    except Captured:
        return captured['geometry']
    finally:
        loader.deduplicate_surfaces = original
    raise AssertionError('Scene did not reach surface deduplication')


def benchmark(scene, config, baseline, repeats):
    geometries = raw_geometry(scene, config)
    timing = {name: {'wall_seconds': [], 'cpu_seconds': []} for name in ('baseline', 'batched')}
    expected = None
    for repeat in range(repeats):
        order = ['baseline', 'batched'] if repeat % 2 == 0 else ['batched', 'baseline']
        for name in order:
            values, warnings = copy.deepcopy(geometries), []
            function = baseline if name == 'baseline' else loader.deduplicate_surfaces
            start, cpu = time.perf_counter(), time.process_time()
            function(values, config, warnings)
            timing[name]['cpu_seconds'].append(time.process_time()-cpu)
            timing[name]['wall_seconds'].append(time.perf_counter()-start)
            # The digest includes ordered triangle bytes, collision triangle
            # bytes, semantic owners and solid/obstacle flags for every geometry.
            signature = (loader.geometry_digest(values), warnings)
            if expected is None:
                expected = signature
            assert signature == expected, 'Geometry/collision/ownership/warnings changed'
            print(scene.parent.name, name, repeat, timing[name]['wall_seconds'][-1], flush=True)
    medians = {name: {kind: statistics.median(samples) for kind, samples in measurements.items()}
               for name, measurements in timing.items()}
    return {'scene': str(scene.resolve()), 'scene_sha256': hashlib.sha256(scene.read_bytes()).hexdigest(),
            'input_geometry_sha256': loader.geometry_digest(geometries),
            'output_geometry_sha256': expected[0], 'warnings': expected[1],
            'input_triangles': sum(len(g.triangles) for g in geometries),
            'exact_geometry_collision_metadata_parity': True,
            'timing': timing, 'medians': medians,
            'median_wall_reduction_percent': 100*(1-medians['batched']['wall_seconds']/medians['baseline']['wall_seconds'])}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scene', action='append', type=Path, required=True)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--baseline-loader', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--repeats', type=int, default=3)
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error('repeats must be positive')
    name = 'spacewatch3d.tasks.task_5b_keyframes._loader_baseline'
    spec = importlib.util.spec_from_file_location(name, args.baseline_loader)
    baseline = importlib.util.module_from_spec(spec)
    sys.modules[name] = baseline
    spec.loader.exec_module(baseline)
    config = Config.from_dict(json.loads(args.config.read_text()))
    report = {'scope': 'local CPU actual-mesh deduplication; not full planner or server wall time',
              'baseline_loader_sha256': hashlib.sha256(args.baseline_loader.read_bytes()).hexdigest(),
              'loader_sha256': hashlib.sha256(Path(loader.__file__).read_bytes()).hexdigest(),
              'cpu_affinity': sorted(os.sched_getaffinity(0)) if hasattr(os, 'sched_getaffinity') else None,
              'config': json.loads(args.config.read_text()), 'repeats': args.repeats,
              'scenes': {scene.parent.name: benchmark(scene, config, baseline.deduplicate_surfaces, args.repeats)
                         for scene in args.scene}}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
