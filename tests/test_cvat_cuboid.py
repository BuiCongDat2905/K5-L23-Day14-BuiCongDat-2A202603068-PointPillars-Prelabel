"""CVAT cuboid points: the server accepts only 16 values, and they must read back to the same box."""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import vf_common as vf  # noqa: E402
from vf_cvat import cuboid_points  # noqa: E402


class TestCvatCuboid(unittest.TestCase):
    def test_cuboid_has_16_values_and_round_trips(self):
        box = {"x": 3.5, "y": -12.0, "z": 0.8, "length": 4.4, "width": 1.9, "height": 1.6, "yaw": 1.2}
        pts = cuboid_points(box)
        self.assertEqual(len(pts), 16)
        self.assertEqual(pts[9:], [0.0] * 7)
        back = vf.cuboid_from_cvat(pts, "Car")
        for k in box:
            self.assertLess(abs(back[k] - box[k]), 1e-9, k)


if __name__ == "__main__":
    unittest.main()
