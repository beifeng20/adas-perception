import cv2
import numpy as np

from adas.alerts import forward_collision, lane_offset
from adas.lanes import LaneDetector, Line
from adas.objects import Detection

W, H = 960, 540


def synthetic_road(shift: int = 0) -> np.ndarray:
    """Dark road with a white left line and yellow right line converging to the horizon.

    `shift` moves only the near (bottom) ends: when the car drifts sideways the
    vanishing point stays roughly in place.
    """
    img = np.full((H, W, 3), 60, np.uint8)
    cv2.line(img, (200 + shift, H), (440, int(H * 0.62)), (255, 255, 255), 8)
    cv2.line(img, (800 + shift, H), (520, int(H * 0.62)), (0, 210, 230), 8)
    return img


def test_detects_both_lines():
    left, right = LaneDetector().detect(synthetic_road())
    assert left is not None and right is not None
    assert abs(left.x_at(H) - 200) < 20
    assert abs(right.x_at(H) - 800) < 20


def test_offset_sign():
    # lane moved left in the image -> car is right of lane center -> positive offset
    left, right = LaneDetector().detect(synthetic_road(shift=-100))
    # expected (480 - 400) / 600 = +0.13
    assert abs(lane_offset(left, right, W, H) - 0.13) < 0.03


def test_keeps_last_line_when_missing():
    det = LaneDetector()
    det.detect(synthetic_road())
    left, right = det.detect(np.full((H, W, 3), 60, np.uint8))  # empty frame
    assert left is not None and right is not None


def test_forward_collision_only_in_ego_lane():
    det = LaneDetector()
    det.detect(synthetic_road())
    poly = det.lane_polygon(H)
    near_in_lane = Detection("car", 0.9, (420, 380, 560, 500))
    near_other_lane = Detection("car", 0.9, (0, 380, 120, 500))
    far_in_lane = Detection("car", 0.9, (470, 340, 500, 360))
    danger = forward_collision([near_in_lane, near_other_lane, far_in_lane], poly, H)
    assert danger == [near_in_lane]


def test_line_eval():
    assert Line(0.5, 10).x_at(100) == 60
