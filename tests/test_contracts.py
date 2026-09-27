"""Guard the boundaries most likely to break during parallel development."""

import contextlib
import io
import json
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from jsonschema import Draft202012Validator

from spacewatch3d.cli import check_inputs, main
from spacewatch3d.contracts import ContractError, read_json, schema, validate_artifact
from spacewatch3d.registry import TASKS
from spacewatch3d.task_api import BackendNotImplemented


FIXTURES = Path(__file__).resolve().parents[1] / 'examples' / 'fixtures'


class ContractTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name) / 'fixtures'
        shutil.copytree(FIXTURES, self.root)

    def edit(self, relative, callback):
        path = self.root / relative
        data = read_json(path)
        callback(data)
        path.write_text(json.dumps(data), encoding='utf-8')
        return path

    def test_all_task_examples_have_valid_upstream_files(self):
        Draft202012Validator.check_schema(schema())
        for task, spec in TASKS.items():
            with self.subTest(task=task):
                check_inputs(task, {k: self.root/v for k,v in spec.fixture_inputs.items()}, check_files=True)
                artifact = validate_artifact(self.root/spec.fixture_output, check_files=True)
                self.assertEqual(spec.output, artifact['kind'])

    def test_task_1a_rejects_panorama_input(self):
        with self.assertRaisesRegex(ContractError, 'perspective'):
            check_inputs('1a', {'video': self.root/'panorama/video.json'}, check_files=True)

    def test_task_5a_requires_different_captures_of_same_scene(self):
        inputs = {name:self.root/'before/objects.json' for name in ('before','after')}
        with self.assertRaisesRegex(ContractError, 'different captures'):
            check_inputs('5a', inputs, check_files=True)

    def test_cross_scene_reference_is_rejected(self):
        path = self.edit('before/objects.json', lambda x: x.update(scene_id='wrong_room'))
        with self.assertRaisesRegex(ContractError, 'scene_id mismatch'):
            validate_artifact(path)

    def test_capture_id_cannot_be_silently_changed(self):
        path = self.edit('before/instances.json', lambda x: x.update(capture_id='wrong_capture'))
        with self.assertRaisesRegex(ContractError, 'capture_id mismatch'):
            validate_artifact(path)

    def test_reordered_point_labels_are_rejected(self):
        path = self.root/'before/labels.csv'
        lines = path.read_text().splitlines()
        lines[1], lines[2] = lines[2], lines[1]
        path.write_text('\n'.join(lines)+'\n')
        with self.assertRaisesRegex(ContractError, 'point order'):
            validate_artifact(self.root/'before/instances.json', check_files=True)

    def test_unassigned_points_are_valid(self):
        self.edit('before/instances.json', lambda x: x['instances'][0].update(point_count=7))
        path = self.root/'before/labels.csv'
        lines = path.read_text().splitlines()
        lines[-1] = '7,'
        path.write_text('\n'.join(lines)+'\n')
        validate_artifact(self.root/'before/instances.json', check_files=True)

    def test_camera_must_refer_to_an_existing_frame(self):
        path = self.edit('before/reconstruction.json', lambda x: x['cameras'][0].update(frame_id='missing'))
        with self.assertRaisesRegex(ContractError, 'unknown frame_id'):
            validate_artifact(path)

    def test_duplicate_instance_ids_are_rejected(self):
        path = self.edit('before/instances.json', lambda x: x['instances'].append(x['instances'][0].copy()))
        with self.assertRaisesRegex(ContractError, 'Duplicate instance_id'):
            validate_artifact(path)

    def test_synthetic_provenance_is_preserved(self):
        path = self.edit('before/objects.json', lambda x: x.update(synthetic=False))
        with self.assertRaisesRegex(ContractError, 'Synthetic input'):
            validate_artifact(path)

    def test_registration_failure_cannot_claim_motion(self):
        path = self.edit('changes.json', lambda x: x['registration'].update(status='failed', baseline_from_current=None))
        with self.assertRaisesRegex(ContractError, 'cannot assert changes'):
            validate_artifact(path)

    def test_registration_failure_allows_unknown_result(self):
        def change(data):
            data['registration'].update(status='failed', baseline_from_current=None)
            data['changes'][0].update(type='unknown', displacement_in_baseline=None)
        path = self.edit('changes.json', change)
        validate_artifact(path, check_files=True)

    def test_change_cannot_reference_missing_object(self):
        path = self.edit('changes.json', lambda x: x['changes'][0].update(after_instance_id='not_found'))
        with self.assertRaisesRegex(ContractError, 'Unknown after_instance_id'):
            validate_artifact(path)

    def test_fixture_motion_matches_known_centroids(self):
        before = read_json(self.root/'before/objects.json')['objects'][0]
        after = read_json(self.root/'after/objects.json')['objects'][0]
        changes = read_json(self.root/'changes.json')
        self.assertNotEqual(before['instance_id'], after['instance_id'])
        matrix = changes['registration']['baseline_from_current']
        point = after['centroid']+[1]
        transformed = [sum(a*b for a,b in zip(row, point)) for row in matrix[:3]]
        for actual, previous, displacement in zip(transformed, before['centroid'], changes['changes'][0]['displacement_in_baseline']):
            self.assertAlmostEqual(actual-previous, displacement)

    def test_keyframe_timestamp_must_match_source(self):
        path = self.edit('before/keyframes.json', lambda x: x['keyframes'][0].update(timestamp_s=999))
        with self.assertRaisesRegex(ContractError, 'timestamp_s'):
            validate_artifact(path)

    def test_missing_file_is_detected(self):
        (self.root/'before/object.obj').unlink()
        with self.assertRaisesRegex(ContractError, 'Missing asset'):
            validate_artifact(self.root/'before/objects.json', check_files=True)

    def test_nan_and_infinity_are_rejected(self):
        for value in ('NaN', '1e999'):
            with self.subTest(value=value):
                path = self.root/'before/reconstruction.json'
                data = read_json(FIXTURES/'before/reconstruction.json')
                data['coordinates']['scale_to_meters'] = 'REPLACE'
                path.write_text(json.dumps(data).replace('"REPLACE"',value))
                with self.assertRaisesRegex(ContractError, 'Non-finite'):
                    validate_artifact(path)

    def test_reference_cycle_is_reported(self):
        path = self.edit('before/frames.json', lambda x: x.update(source_manifest='frames.json'))
        with self.assertRaisesRegex(ContractError, 'Cyclic'):
            validate_artifact(path)

    def test_unimplemented_backend_is_not_reported_as_success(self):
        output = self.root/'new-output'
        errors = io.StringIO()
        with patch('spacewatch3d.tasks.task_4_object_export.pipeline.run', side_effect=BackendNotImplemented('test backend missing')):
            with contextlib.redirect_stderr(errors):
                status = main(['run','--task','4','--input',f'instances={self.root / "before/instances.json"}','--output',str(output)])
        self.assertEqual(status, 2)
        self.assertIn('test backend missing', errors.getvalue())
        self.assertFalse(output.exists())


if __name__ == '__main__':
    unittest.main()
