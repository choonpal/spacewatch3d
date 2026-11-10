"""Exact keys/inverse and downstream payload parity for integer voxel grouping."""
import importlib.util
from pathlib import Path
import unittest
import numpy as np

SOURCE = Path(__file__).resolve().parents[1]/'vendor/panovggt_pipeline_v2/workers/voxel_keys.py'
spec = importlib.util.spec_from_file_location('voxel_keys', SOURCE)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class VoxelKeysTests(unittest.TestCase):
    def check_parity(self, keys):
        before = keys.copy()
        expected, expected_inverse = np.unique(keys, axis=0, return_inverse=True)
        actual, inverse = module.unique_voxel_keys(keys)
        np.testing.assert_array_equal(actual, expected)
        np.testing.assert_array_equal(inverse, expected_inverse.reshape(-1))
        np.testing.assert_array_equal(keys, before)
        self.assertEqual(inverse.dtype, np.dtype(np.intp))
        np.testing.assert_array_equal(actual[inverse], keys)

    def test_empty_singleton_and_equal_keys(self):
        for n in [0, 1, 100]:
            self.check_parity(np.tile(np.array([[-9, 0, 7]], dtype=np.int64), (n, 1)))

    def test_signed_extremes_without_packed_key_overflow(self):
        lo, hi = np.iinfo(np.int64).min, np.iinfo(np.int64).max
        keys = np.array([[lo, hi, 0], [hi, lo, -1], [lo, hi, 0], [lo, lo, lo], [hi, hi, hi], [0, 0, 0]], dtype=np.int64)
        self.check_parity(keys)

    def test_noncontiguous_and_readonly_inputs(self):
        data = np.arange(600, dtype=np.int64).reshape(100, 6)[::-2, ::2]
        data.flags.writeable = False
        self.check_parity(data)

    def test_randomized_mixed_duplicate_density(self):
        rng = np.random.default_rng(20261026)
        for n in [0, 1, 2, 17, 1000, 12000]:
            for extent in [1, 4, 1000000]:
                for _ in range(3):
                    self.check_parity(rng.integers(-extent, extent+1, size=(n, 3), dtype=np.int64))

    def test_bincount_preserves_floating_summation_and_color_rounding(self):
        rng = np.random.default_rng(11)
        keys = rng.integers(-4, 5, size=(20000, 3), dtype=np.int64)
        _, ref = np.unique(keys, axis=0, return_inverse=True)
        _, inv = module.unique_voxel_keys(keys)
        ref = ref.reshape(-1)
        values = rng.normal(size=len(keys))*rng.choice([1e-12, 1.0, 1e12], len(keys))
        colors = rng.integers(0, 256, size=len(keys))
        for data in [values, colors]:
            a = np.bincount(ref, weights=data)/np.bincount(ref)
            b = np.bincount(inv, weights=data)/np.bincount(inv)
            self.assertEqual(a.tobytes(), b.tobytes())
            self.assertEqual(np.rint(a).tobytes(), np.rint(b).tobytes())

    def test_rejects_non_integer_or_non_xyz_rows(self):
        for data in [np.zeros((3, 3), np.float64), np.zeros((3, 3), np.int32)]:
            with self.assertRaises(TypeError): module.unique_voxel_keys(data)
        for data in [np.zeros(3, np.int64), np.zeros((3, 2), np.int64)]:
            with self.assertRaises(ValueError): module.unique_voxel_keys(data)


if __name__ == '__main__': unittest.main()
