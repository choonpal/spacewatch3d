"""Small command line interface for contract checks and task entry points."""

import argparse
import importlib
import json
import sys
from pathlib import Path

from .contracts import ContractError, read_json, validate_artifact
from .registry import TASKS
from .task_api import BackendNotImplemented, TaskRequest


def check_inputs(task: str, inputs: dict[str, Path], *, check_files: bool) -> dict:
    spec = TASKS[task]
    if set(inputs) != set(spec.inputs):
        raise ContractError(f'Task {task} input names must be {list(spec.inputs)}')
    manifests = {name: validate_artifact(path, check_files=check_files) for name, path in inputs.items()}
    for name, kind in spec.inputs.items():
        if manifests[name]['kind'] != kind:
            raise ContractError(f'{name} requires a {kind} manifest')
    if task in ('1a', '1b'):
        expected = 'perspective' if task == '1a' else 'equirectangular'
        if manifests['video']['projection'] != expected:
            raise ContractError(f'Task {task} requires {expected} input')
    if task == '5a':
        before, after = manifests['before'], manifests['after']
        if before['scene_id'] != after['scene_id'] or before['capture_id'] == after['capture_id']:
            raise ContractError('Task 5a needs different captures of the same scene')
    return manifests


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog='spacewatch3d')
    commands = parser.add_subparsers(dest='command', required=True)
    commands.add_parser('tasks', help='List task boundaries')
    validate = commands.add_parser('validate', help='Validate a manifest and its upstream references')
    validate.add_argument('manifest', type=Path)
    validate.add_argument('--check-files', action='store_true')
    fixtures = commands.add_parser('check-fixtures', help='Check synthetic contracts, without running any models')
    fixtures.add_argument('--task', choices=TASKS)
    fixtures.add_argument('--fixtures', type=Path, default=Path('examples/fixtures'))
    run = commands.add_parser('run', help='Call a task backend (initial backends are unimplemented)')
    run.add_argument('--task', choices=TASKS, required=True)
    run.add_argument('--input', action='append', required=True, metavar='NAME=MANIFEST')
    run.add_argument('--output', type=Path, required=True)
    run.add_argument('--config', type=Path)
    args = parser.parse_args(argv)
    try:
        if args.command == 'tasks':
            for name, spec in TASKS.items():
                print(f'{name:>2}  {spec.title} | {spec.package} | {spec.inputs} -> {spec.output}')
        elif args.command == 'validate':
            data = validate_artifact(args.manifest, check_files=args.check_files)
            print(f'OK: {data["kind"]} {data["scene_id"]}/{data["capture_id"]}; synthetic={data["synthetic"]}')
        elif args.command == 'check-fixtures':
            for name in ([args.task] if args.task else TASKS):
                spec = TASKS[name]
                inputs = {key: args.fixtures / relative for key, relative in spec.fixture_inputs.items()}
                check_inputs(name, inputs, check_files=True)
                data = validate_artifact(args.fixtures / spec.fixture_output, check_files=True)
                if data['kind'] != spec.output or not data['synthetic']:
                    raise ContractError(f'Task {name} requires a synthetic {spec.output} fixture')
                print(f'OK: task {name} synthetic input/output contracts (no model execution)')
        else:
            inputs = {}
            for value in args.input:
                name, separator, path = value.partition('=')
                if not separator or not path or name in inputs:
                    raise ContractError('Each --input must have a unique NAME=MANIFEST value')
                inputs[name] = Path(path).resolve()
            input_data = check_inputs(args.task, inputs, check_files=True)
            config = read_json(args.config) if args.config else {}
            if args.output.exists() and (not args.output.is_dir() or any(args.output.iterdir())):
                raise ContractError('Use a new or empty output directory for each task run')
            module = importlib.import_module(f'spacewatch3d.tasks.{TASKS[args.task].package}.pipeline')
            result = module.run(TaskRequest(inputs, args.output.resolve(), config))
            result = Path(result).resolve()
            if not result.is_relative_to(args.output.resolve()):
                raise ContractError('The returned manifest must be inside the output directory')
            data = validate_artifact(result, check_files=True)
            source = input_data['after'] if args.task == '5a' else next(iter(input_data.values()))
            if data['kind'] != TASKS[args.task].output:
                raise ContractError('Backend returned the wrong artifact kind')
            if (data['scene_id'], data['capture_id']) != (source['scene_id'], source['capture_id']):
                raise ContractError('Backend output scene/capture differs from its input')
            if any(item['synthetic'] for item in input_data.values()) and not data['synthetic']:
                raise ContractError('Synthetic inputs must remain marked synthetic')
            print(result)
        return 0
    except (ContractError, BackendNotImplemented, OSError, json.JSONDecodeError) as exc:
        print(f'ERROR: {exc}', file=sys.stderr)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
