"""ERP human boxes must wrap across the seam rather than erase the middle."""
import sys
import unittest
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / 'tools/scenediff_mesh'))
from build_focus_comparison import wrapped_box_mask


class PersonBoxProjection(unittest.TestCase):
    def test_unwrapped_positive_and_negative_boxes_cover_both_seam_edges(self):
        expected = np.zeros((4, 8), bool)
        expected[1:3, [0, 1, 6, 7]] = True
        np.testing.assert_array_equal(wrapped_box_mask((4, 8), [6, 1, 10, 3]), expected)
        np.testing.assert_array_equal(wrapped_box_mask((4, 8), [-2, 1, 2, 3]), expected)

    def test_pole_clamping_and_full_circle_do_not_create_duplicate_pixels(self):
        result = wrapped_box_mask((4, 8), [-10, -3, 12, 2])
        self.assertEqual(int(result.sum()), 16)
        self.assertTrue(result[:2].all())
        self.assertFalse(result[2:].any())

    def test_empty_or_outside_vertical_interval_excludes_nothing(self):
        self.assertFalse(wrapped_box_mask((4, 8), [1, 2, 1, 4]).any())
        self.assertFalse(wrapped_box_mask((4, 8), [1, 5, 3, 8]).any())


if __name__ == '__main__':
    unittest.main()
