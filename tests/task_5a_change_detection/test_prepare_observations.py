import copy
import json
import tempfile
import unittest
from pathlib import Path

from spacewatch3d.tasks.task_5a_change_detection.prepare_observations import extract, main


class ObservationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.source = Path(self.temp.name) / 'source.json'
        self.data = {
            'name': 'scene0462_00', 'num_points': 3, 'num_masks': 2,
            'positions': [[0, 0, 0], [2, 4, 6], [1, 1, 1]],
            'colors': [[0, 32, 255], [255, 32, 0], [128, 128, 128]],
            'masks': [[0, 1], [1, 2]], 'labels': ['box', 'box'], 'scores': [0.1, 0.2],
        }

    def extract(self, data=None):
        self.source.write_text(json.dumps(data or self.data), encoding='utf-8')
        return extract(self.source)

    def test_geometry_and_color_descriptor_from_input_points(self):
        obj = self.extract()['observations'][0]
        self.assertEqual(obj['centroid'], [1, 2, 3])
        self.assertEqual(obj['volume'], 48)
        self.assertEqual(obj['point_count'], 2)
        self.assertEqual(obj['feature'][0], 0.5)
        self.assertEqual(obj['feature'][7], 0.5)
        self.assertEqual(obj['feature'][9], 1)
        for channel in range(3):
            self.assertAlmostEqual(sum(obj['feature'][channel * 8:(channel + 1) * 8]), 1)

    def test_overlap_is_audited_without_inventing_exclusive_instances(self):
        result = self.extract()
        self.assertEqual(result['audit']['points_in_multiple_masks'], 1)
        self.assertEqual(result['audit']['unique_masked_points'], 3)
        self.assertEqual(sum(o['point_count'] for o in result['observations']), 4)
        self.assertFalse(result['coverage_complete'])
        self.assertFalse(result['provenance']['source_matches_task4'])
        self.assertIsNone(result['synthetic'])
        self.assertIsNone(result['provenance']['source_url'])
        self.assertEqual(result['alignment']['status'], 'single_capture')

    def test_invalid_or_duplicate_mask_indices_are_rejected(self):
        for indices in ([0, 3], [0, 0], [-1], [True], [0.5]):
            data = copy.deepcopy(self.data)
            data['masks'][0] = indices
            with self.subTest(indices=indices), self.assertRaisesRegex(ValueError, 'indices'):
                self.extract(data)

    def test_nonfinite_coordinates_or_invalid_colors_are_rejected(self):
        for key, value in (('positions', [float('nan'), 0, 0]), ('colors', [256, 0, 0])):
            data = copy.deepcopy(self.data)
            data[key][0] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                self.extract(data)

    def test_count_mismatch_is_rejected(self):
        data = copy.deepcopy(self.data)
        data['num_points'] += 1
        with self.assertRaisesRegex(ValueError, 'counts'):
            self.extract(data)

    def test_empty_mask_is_recorded(self):
        data = copy.deepcopy(self.data)
        data['masks'][0] = []
        result = self.extract(data)
        self.assertEqual(result['audit']['empty_mask_ids'], [0])
        self.assertEqual(result['audit']['observation_count'], 1)

    def test_existing_output_is_not_overwritten(self):
        self.extract()
        output = Path(self.temp.name) / 'observations.json'
        output.write_text('preserve', encoding='utf-8')
        with self.assertRaises(FileExistsError):
            main(['--source', str(self.source), '--output', str(output)])
        self.assertEqual(output.read_text(), 'preserve')

    def test_cli_rejects_modified_source_without_mislabeling_it_as_real_dataset(self):
        self.extract()
        output = Path(self.temp.name) / 'observations.json'
        with self.assertRaisesRegex(ValueError, 'SHA-256'):
            main(['--source', str(self.source), '--output', str(output)])
        self.assertFalse(output.exists())


if __name__ == '__main__':
    unittest.main()
