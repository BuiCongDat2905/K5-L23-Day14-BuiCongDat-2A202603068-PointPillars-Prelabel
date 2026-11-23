"""CVAT cuboid points: the server accepts only 16 values, and they must read back to the same box."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import vf_common as vf  # noqa: E402
from vf_cvat import cuboid_points  # noqa: E402


def test_cuboid_has_16_values_and_round_trips():
    box = {"x": 3.5, "y": -12.0, "z": 0.8, "length": 4.4, "width": 1.9, "height": 1.6, "yaw": 1.2}
    pts = cuboid_points(box)
    assert len(pts) == 16
    assert pts[9:] == [0.0] * 7
    back = vf.cuboid_from_cvat(pts, "Car")
    for k in box:
        assert abs(back[k] - box[k]) < 1e-9, k
