from pathlib import Path
import sys
import tempfile
import unittest
import numpy as np
from plyfile import PlyData

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
from common import read, digest, validate_config
from component_filter import DTYPE, header, filter_cloud
from slab_components import self_check


class ComponentTests(unittest.TestCase):
    def test_slab_labels_equal_global_dense_connectivity(self):
        self.assertEqual(self_check()['dense_equivalence_cases'], 4)

    def test_exact_threshold_and_unchanged_point_payload(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d); source = root/'source.ply'; target = root/'filtered.ply'
            sizes = [9999, 10000, 10001]
            data = np.zeros(sum(sizes), dtype=DTYPE)
            begin = 0
            for i, count in enumerate(sizes):
                data['x'][begin:begin+count] = i*10
                data['red'][begin:begin+count] = np.arange(count) % 256
                begin += count
            with source.open('wb') as f:
                f.write(header(len(data))); data.tofile(f)
            options = read(HERE/'configs/dorm2.json')['postprocessing']['connected_components']
            options = {**options, 'chunk_points': 1234, 'slab_width': 3}
            report = filter_cloud(source, target, root/'audit', options, digest(source), progress=lambda _: None)
            self.assertEqual(report['kept_points'], 10001)
            self.assertEqual(report['removed_points'], 19999)
            self.assertEqual(report['kept_components'], 1)
            self.assertEqual(report['largest_removed_component'], 10000)
            actual = PlyData.read(target)['vertex'].data
            self.assertEqual(actual.tobytes(), data[-10001:].tobytes())
            self.assertTrue(report['validation']['source_unchanged'])

    def test_no_silent_ror_and_invalid_component_settings(self):
        c = read(HERE/'configs/dorm2.json')
        c['postprocessing']['radius_outlier_removal'] = True
        with self.assertRaises(ValueError):
            validate_config(c)
        for value in [-1, 0.1, True]:
            c = read(HERE/'configs/dorm2.json')
            c['postprocessing']['connected_components']['remove_max_points_inclusive'] = value
            with self.assertRaises(ValueError):
                validate_config(c)


if __name__ == '__main__':
    unittest.main()
