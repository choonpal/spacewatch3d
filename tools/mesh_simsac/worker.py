"""Run fresh RGB inference and publish a job-specific interactive result."""
import argparse
from pathlib import Path
import sys
import time

from .core import read, sha, write


def execute(manifest, output, runtime):
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    started = time.time()

    def stage(name):
        write(output/'status.json', dict(state='running', stage=name, started_unix=started))

    try:
        for path in runtime.get('dependency_paths', []):
            sys.path.insert(0, path)
        stage('input_validation')
        original = read(manifest)
        for field, expected in runtime.get('model_hashes', {}).items():
            if sha(runtime[field]) != expected:
                raise ValueError(f'Model changed after job submission: {field}')
        checkpoint = runtime.get('person_checkpoint')
        if not checkpoint and not all('person_mask' in f for phase in ['before', 'after'] for f in original[phase]['frames']):
            raise ValueError('촬영자 제외를 위해 사람 검출 모델 또는 모든 RGB의 person_mask가 필요합니다.')
        from .prepare import prepare
        from .run import run
        from .appearance import extract
        from .classify import classify
        from .validate import validate
        from .preview import export
        from .build_event_viewer import build
        stage('registration')
        root = output/'detection'
        prepare(manifest, root, capturer_checkpoint=checkpoint, device=runtime.get('device','cuda'))
        stage('simsac')
        run(root, runtime['simsac_source'], runtime['simsac_checkpoint'], checkpoint, runtime.get('device', 'cuda'))
        stage('classification')
        extract(root, runtime['simsac_source'], runtime['simsac_checkpoint'], runtime.get('device', 'cuda'))
        classify(root, manifest)
        validate(root)
        stage('viewer')
        export(root, manifest, runtime['three_package'])
        build(root, manifest)
        result = read(root/'viewer-data.json')
        result['labels'] = [original[p]['id'] for p in ['before', 'after']]
        result['job_provenance'] = dict(fresh_inference=True, input_source='uploaded_original_rgb',
                                      gaussian_inputs_used=False, model=read(root/'results/model.json')['checkpoint_sha256'])
        from .observation_quality import quality_settings
        result['job_provenance']['object_quality']=quality_settings(original.get('settings',{}))
        result['job_provenance']['reference_vote']='Strict majority of usable references; ties are not positive'
        write(root/'viewer-data.json', result)
        write(output/'status.json', dict(state='complete', stage='complete', counts=result['counts'],
                                        inference_pairs=result['inference_pairs'], seconds=time.time()-started))
    except Exception as error:
        write(output/'status.json', dict(state='failed', stage='failed', message=str(error)))
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--runtime', type=Path, required=True)
    args = parser.parse_args()
    execute(args.manifest, args.output, read(args.runtime))
