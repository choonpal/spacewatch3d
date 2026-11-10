#!/usr/bin/env python3
"""One synchronous, restartable entry point for the complete configured reconstruction."""
from pathlib import Path
import argparse
import fcntl
import importlib.metadata
import os
import shutil
import subprocess
import sys
import time
import traceback
from common import read, save, record, verify, digest, object_hash, validate_config, make_runtime, cache_contract

HERE = Path(__file__).resolve().parent
OBS_FILES = ['observations.npz', 'initial_pair_transforms.json', 'edge_audit.json',
             'observation_summary.json', 'observation_inputs.json', 'bridge_pairs.json']


class Executor:
    def __init__(self, root, identity, command_runner=subprocess.run):
        self.root = root
        self.identity = identity
        self.command_runner = command_runner
        self.chain = identity
        self.receipts = []

    def stage(self, name, command, outputs, snapshot=None, clean_partial=()):
        receipt_path = self.root/'stages'/f'{name}.json'
        stage_input = object_hash([self.identity, self.chain, name, command])
        if receipt_path.exists():
            receipt = read(receipt_path)
            if receipt['input_identity'] != stage_input:
                raise ValueError(f'Stage dependency mismatch: {name}; use a new output directory')
            for artifact in receipt['artifacts']:
                verify(artifact)
            resumed = True
        else:
            # Incomplete matching/export attempts cannot donate unreceipted output to a retry.
            for relative in clean_partial:
                p = self.root/relative
                if p.exists():
                    backup = self.root/'interrupted'/f'{name}_{time.time_ns()}'/relative
                    backup.parent.mkdir(parents=True, exist_ok=True)
                    shutil.move(str(p), str(backup))
            save(self.root/'status.json', {'status': 'running', 'stage': name, 'pid': os.getpid()})
            started = time.perf_counter()
            log = self.root/'logs'/f'{name}.log'
            log.parent.mkdir(parents=True, exist_ok=True)
            with log.open('w') as stream:
                self.command_runner(command, stdout=stream, stderr=subprocess.STDOUT, check=True,
                                    env={**os.environ, 'PANOVGGT_RUN_ROOT': str(self.root)}, cwd=HERE)
            artifacts = []
            for relative in outputs:
                p = self.root/relative
                if not p.exists():
                    raise RuntimeError(f'Stage {name} did not create {relative}')
                if snapshot:
                    q = self.root/snapshot/relative
                    q.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(p, q)
                    artifacts.append(record(q))
                elif p.is_dir():
                    artifacts.extend(record(f) for f in sorted(p.rglob('*')) if f.is_file())
                else:
                    artifacts.append(record(p))
            receipt = {'status': 'complete', 'name': name, 'input_identity': stage_input,
                       'elapsed_seconds': time.perf_counter()-started, 'command': command, 'artifacts': artifacts}
            save(receipt_path, receipt)
            resumed = False
        if snapshot:
            # Restore this phase's graph before planning the following recovery stage, including on resume.
            for relative in outputs:
                shutil.copyfile(self.root/snapshot/relative, self.root/relative)
        self.chain = object_hash(receipt)
        self.receipts.append({'name': name, 'resumed': resumed, 'seconds': receipt['elapsed_seconds']})
        print(('RESUMED' if resumed else 'COMPLETE'), name, flush=True)
        return receipt


def code_identity():
    paths = sorted(HERE.glob('*.py')) + sorted((HERE/'workers').glob('*.py')) + sorted((HERE/'viewer').glob('*'))
    return {str(p.relative_to(HERE)): digest(p) for p in paths if p.is_file()}


def environment():
    versions = {'python': sys.version.split()[0]}
    for name in ['torch', 'numpy', 'scipy', 'kornia', 'opencv-python', 'opencv-python-headless',
                 'Pillow', 'plyfile', 'matplotlib', 'omegaconf', 'einops']:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def run(config_path, output, replay=None, validate_only=False):
    c = read(config_path)
    validate_config(c)
    bundle = read(c['input']['bundle'])
    if bundle['source']['sha256'] != c['input']['source_sha256']:
        raise ValueError('Configuration is bound to a different video')
    items = [bundle['source'], bundle['frame_manifest'], *bundle['frames'].values(), *bundle['base_masks'].values()]
    for item in items:
        verify(item)
    replay_data = None
    if replay:
        replay_data = read(replay)
        if replay_data['cache_contract'] != cache_contract(c):
            raise ValueError('Replay model/mask/matching settings differ; cannot reuse these caches')
        if replay_data['input_bundle_sha256'] != digest(c['input']['bundle']):
            raise ValueError('Replay input bundle differs')
        for item in replay_data['files'].values():
            verify(item)
    if validate_only:
        print('VALIDATED INPUTS', len(bundle['frames']), 'frames; mode:', 'CPU cache replay' if replay else 'fresh server inference')
        return
    root = Path(output).resolve()
    root.mkdir(parents=True, exist_ok=True)
    lock = (root/'.run.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if not replay:
        gpu_lock = open(f'/tmp/panovggt-pipeline-gpu-{c["execution"]["physical_gpu"]}.lock', 'a')
        fcntl.flock(gpu_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    identity = {'configuration': c, 'bundle': bundle, 'replay': replay_data,
                'code': code_identity(), 'environment': environment()}
    signature = object_hash(identity)
    if (root/'run_identity.json').exists():
        if read(root/'run_identity.json')['sha256'] != signature:
            raise ValueError('Run inputs/configuration/code/environment changed; select a new output directory')
    elif any(p.name != '.run.lock' for p in root.iterdir()):
        raise ValueError('Output directory is not empty and has no run identity; refusing to overwrite')
    else:
        save(root/'run_identity.json', {'sha256': signature, **identity})
        save(root/'config.json', make_runtime(c, root))
        if replay_data:
            save(root/'replay_inventory.json', replay_data)
    threads = str(c['execution']['cpu_threads'])
    os.environ.update(OMP_NUM_THREADS=threads, OPENBLAS_NUM_THREADS=threads,
                      CUDA_VISIBLE_DEVICES=str(c['execution']['physical_gpu']))
    executor = Executor(root, signature)
    def worker(filename, *args):
        return [sys.executable, '-u', str(HERE/'workers'/filename), *args]
    started = time.perf_counter()
    try:
        executor.stage('inputs', worker('prepare_inputs.py'),
                       ['input', 'masks', 'reviewed_mask_manifest.json', 'mask_review.json', 'input_identity.json'])
        executor.stage('inference', worker('infer.py'), ['windows', 'inference_status.json'])
        executor.stage('depth', worker('refine.py', 'prepare'),
                       ['depths.npy', 'valid.npy', 'prepared.npz', 'source_hashes.json', 'depth_calibration.json'])
        sources = []
        def observe(phase):
            save(root/'matching_sources_ready.json', {'sources': sources.copy(),
                 'completed_match_stage_receipts': [x for x in executor.receipts if x['name'].startswith('match_')]})
            executor.stage('observations_'+phase, worker('refine.py', 'observations'),
                           OBS_FILES, snapshot='observation_phases/'+phase)
        executor.stage('match_base', worker('match.py'), ['matches', 'matching_summary.json'], clean_partial=['matches'])
        sources.append('matches')
        observe('base')
        if c['matching']['cross_face'] and not read(root/'observation_summary.json')['connected']:
            executor.stage('match_cross', worker('match.py', '--bridge'),
                           ['bridge_matches', 'bridge_matching_summary.json'], clean_partial=['bridge_matches'])
            sources.append('bridge_matches')
            observe('cross')
        if c['matching']['sift']['enabled'] and not read(root/'observation_summary.json')['connected']:
            executor.stage('match_sift', worker('enhance.py'),
                           ['enhanced_matches', 'enhanced_matching_summary.json'], clean_partial=['enhanced_matches'])
            sources.append('enhanced_matches')
            observe('sift')
        if c['matching']['high_resolution']['enabled'] and not read(root/'observation_summary.json')['connected']:
            executor.stage('plan_hires', worker('plan_hires.py'), ['hires_pairs.json'])
            executor.stage('match_hires', worker('match_hires.py'),
                           ['hires_matches', 'hires_matching_summary.json'], clean_partial=['hires_matches'])
            sources.append('hires_matches')
            # No background process: the preceding stage must have returned and its files must be sealed.
            observe('hires')
        executor.stage('optimize', worker('optimize.py'),
                       ['components', 'components_optimized.npz', 'components_optimization_report.json'], clean_partial=['components'])
        executor.stage('connect', worker('connect.py'), ['optimized.npz', 'registration_report.json', 'unverified_bridges.json'])
        executor.stage('export', worker('export.py'), ['unfiltered_result', 'export_status.json'], clean_partial=['unfiltered_result'])
        executor.stage('filter_components', worker('filter_components.py'),
                       ['result', 'component_filter', 'component_filter_status.json'],
                       clean_partial=['result', 'component_filter'])
        shutil.copytree(HERE/'viewer', root/'viewer', dirs_exist_ok=True)
        executor.stage('viewer', worker('build_viewer.py'), ['viewer'])
        result = read(root/'result/validation.json')
        save(root/'run_report.json', {'status': 'complete', 'mode': 'verified_cache_replay' if replay else 'fresh_inference',
             'invocation_seconds': time.perf_counter()-started, 'stages': executor.receipts,
             'run_identity': signature, 'result': result,
             'reproducibility': 'Checksummed inputs and intermediate data; bitwise equality of fresh GPU inference is not guaranteed'})
        save(root/'status.json', {'status': 'complete', 'point_cloud': result['point_cloud'],
                                'unverified_bridges': result['unverified_bridge_count']})
    except BaseException as ex:
        save(root/'status.json', {'status': 'failed', 'error': str(ex), 'traceback': traceback.format_exc()})
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--output', required=True, help='New run directory; same command resumes an unchanged run')
    parser.add_argument('--replay-cache', help='Explicit sealed caches; never runs GPU inference or falls back to it')
    parser.add_argument('--validate-only', action='store_true')
    args = parser.parse_args()
    run(args.config, args.output, args.replay_cache, args.validate_only)


if __name__ == '__main__':
    main()
