"""Audit the bundled public tour independently of optional local model files."""
import hashlib
import json
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1] / 'assets/gaussian-change'
TOUR = ROOT / 'gaussian-dual-path-tour-20261026'


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


class GaussianRuntime(unittest.TestCase):
    def test_public_runtime_matches_its_integrity_manifest(self):
        manifest = read(ROOT / 'runtime-manifest.json')
        self.assertFalse(manifest['model_parameters_included'])
        for name, meta in manifest['files'].items():
            data = (ROOT / name).read_bytes()
            self.assertEqual(len(data), meta['bytes'], name)
            self.assertEqual(hashlib.sha256(data).hexdigest(), meta['sha256'], name)

    def test_original_cameras_links_and_candidate_inputs_are_present(self):
        data, routes = read(TOUR / 'gaussian-data.json'), read(TOUR / 'tour-routes.json')
        self.assertEqual(len(data['zones']), 26)
        self.assertEqual([(g['capture'], len(g['nodes']), len(g['links']), len(g['segments']))
                          for g in routes['groups']], [('dorm2', 13, 9, 4), ('dorm3', 13, 12, 1)])
        self.assertFalse(routes['new_links_created'])
        for phases in data['renders'].values():
            for render in phases.values():
                self.assertEqual(hashlib.sha256((TOUR / render['image']).read_bytes()).hexdigest(), render['image_sha256'])
        fixed = read(TOUR / 'tour-candidates.json')
        original = read(ROOT / 'gaussian-map-comparison-20261025/simsac-candidates.json')
        self.assertEqual(fixed['boxes'], original['boxes'])
        self.assertFalse(fixed['new_inference'])
        for mode in fixed['boxes'].values():
            for box in mode:
                for render in fixed['source_renders'][box['zone']].values():
                    self.assertTrue((TOUR / fixed['source_asset_prefix'] / render['image']).is_file())


if __name__ == '__main__':
    unittest.main()
