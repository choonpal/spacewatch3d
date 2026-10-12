"""Evidence correctness checks; the demo is independent of common contracts."""
import sys
import tempfile
import unittest
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from build_change_evidence_demo import PLY_DTYPE, read_labeled_ply, spatial_regions, nearest_evidence, render_indices, sample_surface


class EvidenceTests(unittest.TestCase):
    def test_binary_params_preserves_raw_record_order_and_rejects_truncation(self):
        header = ("ply\nformat binary_little_endian 1.0\nelement params 3\n"
                  "property float x\nproperty float y\nproperty float z\n"
                  "property uchar red\nproperty uchar green\nproperty uchar blue\n"
                  "property float cd_type\nend_header\n").encode()
        records = np.zeros(3, PLY_DTYPE)
        records["x"] = [4, 1, 7]
        records["red"] = [2, 5, 8]
        records["cd_type"] = [1, 0, 1]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "source.ply"
            path.write_bytes(header + records.tobytes())
            xyz, rgb, changed = read_labeled_ply(path)
            np.testing.assert_array_equal(xyz[:, 0], [4, 1, 7])
            np.testing.assert_array_equal(rgb[:, 0], [2, 5, 8])
            np.testing.assert_array_equal(np.flatnonzero(changed), [0, 2])
            del xyz, rgb, changed
            path.write_bytes(header + records.tobytes()[:-1])
            with self.assertRaisesRegex(ValueError, "truncated"):
                read_labeled_ply(path)

    def test_regions_preserve_source_indices_and_exclude_unchanged_points(self):
        xyz = np.array([[0, 0, 0], [5, 5, 5], [.01, 0, 0], [.02, 0, 0], [5.01, 5, 5], [1, 1, 1]])
        changed = np.array([1, 1, 0, 1, 1, 1], bool)
        regions = spatial_regions(xyz, changed, .1, 2)
        self.assertEqual({tuple(r) for r in regions}, {(0, 3), (1, 4)})
        self.assertEqual(spatial_regions(xyz, np.zeros(6, bool), .1, 1), [])

    def test_distances_are_geometric_not_object_correspondences(self):
        before = np.array([[0., 0, 0], [2, 0, 0]])
        after = np.array([[2., 0, .5], [0, 0, 0]])
        distances, indices = nearest_evidence(before, after)
        np.testing.assert_array_equal(indices, [1, 0])
        np.testing.assert_allclose(distances, [0, .5])
        with self.assertRaises(ValueError):
            nearest_evidence(before, np.empty((0, 3)))

    def test_render_samples_are_original_points_and_keep_label_boundary(self):
        xyz = np.array([[.001, 0, 0], [.002, 0, 0], [.003, 0, 0], [1, 1, 1]])
        selected = render_indices(xyz, np.array([0, 0, 1, 1]), .1)
        np.testing.assert_array_equal(selected, [0, 2, 3])

    def test_distance_sampling_preserves_raw_indices_and_spatial_extent(self):
        xyz = np.array([[10., 0, 0], [.01, 0, 0], [.02, 0, 0], [20, 0, 0], [999, 0, 0]])
        indices = np.array([0, 1, 2, 3])
        selected = sample_surface(xyz, indices, voxel=.1, limit=2)
        np.testing.assert_array_equal(selected, [1, 3])
        self.assertTrue(np.isin(selected, indices).all())


if __name__ == "__main__":
    unittest.main()
