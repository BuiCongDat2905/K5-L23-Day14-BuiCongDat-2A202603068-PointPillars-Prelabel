"""The lab scores correction cost only for objects within 50 m (BEV) of the LiDAR."""

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from vf_pp.core import MAX_RANGE, correction_cost, within_range  # noqa: E402


def box(x, y, label="Car"):
    return {"x": x, "y": y, "z": 0.0, "length": 4.0, "width": 1.8, "height": 1.5, "yaw": 0.0, "label": label}


def test_max_range_is_50_m():
    assert MAX_RANGE == 50.0


def test_boundary_is_inclusive():
    on_edge = box(30.0, 40.0)            # exactly 50 m
    just_out = box(30.0, 40.01)
    assert within_range([on_edge, just_out]) == [on_edge]


def test_uses_bev_distance_not_x_only():
    assert within_range([box(-45.0, 0.0)])          # behind the car, 45 m
    assert not within_range([box(40.0, -35.0)])     # 53 m diagonally


def test_far_label_is_not_an_add():
    near, far = box(10.0, 0.0), box(60.0, 0.0)
    c = correction_cost([near, far], [box(10.0, 0.0)])
    assert (c["gt"], c["match"], c["add"], c["delete"], c["edit"], c["cost"]) == (1, 1, 0, 0, 0, 0.0)


def test_far_prediction_is_not_a_delete():
    c = correction_cost([box(10.0, 0.0)], [box(10.0, 0.0), box(0.0, 55.0)])
    assert (c["pred"], c["delete"], c["cost"]) == (1, 0, 0.0)


def test_only_far_objects_cost_nothing():
    far = [box(70.0, 0.0), box(0.0, -60.0)]
    assert correction_cost(far, [box(51.0, 0.0)])["cost"] == 0.0


def test_near_errors_still_counted():
    gt = [box(10.0, 0.0), box(20.0, 5.0), box(80.0, 0.0)]
    pred = [box(10.0, 0.0), box(-15.0, -15.0)]       # one hit, one extra, one miss within 50 m
    c = correction_cost(gt, pred)
    assert (c["add"], c["delete"], c["cost"]) == (1, 1, 1.3)


def test_compare_view_hides_far_boxes():
    import pp_compare

    drawn = pp_compare.classify([box(10.0, 0.0), box(60.0, 0.0)], [box(10.0, 0.0), box(0.0, 70.0)])
    assert all(math.hypot(b["x"], b["y"]) <= MAX_RANGE for b, _ in drawn)
    assert [c for _, c in drawn] == [pp_compare.GREEN]
