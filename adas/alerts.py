"""Simple ADAS logic on top of perception: lane departure and forward collision warnings."""
import cv2
import numpy as np

from .lanes import Line
from .objects import Detection

VEHICLES = {"car", "truck", "bus", "motorcycle", "bicycle", "person"}


def lane_offset(left: Line | None, right: Line | None, w: int, h: int) -> float | None:
    """Offset of the camera (image center) from the lane center, as a fraction of lane width.

    0 = centered, +0.5 = on the right line, -0.5 = on the left line.
    Assumes the camera is mounted on the car's center axis.
    """
    if left is None or right is None:
        return None
    xl, xr = left.x_at(h), right.x_at(h)
    width = xr - xl
    if width <= 0:
        return None
    return (w / 2 - (xl + xr) / 2) / width


def lane_departure(offset: float | None, threshold: float = 0.25) -> bool:
    return offset is not None and abs(offset) > threshold


def forward_collision(dets: list[Detection], lane_poly: np.ndarray | None, h: int,
                      min_height_ratio: float = 0.18) -> list[Detection]:
    """Objects in the ego lane that are close (approximated by large bounding-box height).

    Bigger box height ~ smaller distance (pinhole camera model), so the ratio
    box_height / frame_height is used as a cheap proxy for distance.
    """
    if lane_poly is None:
        return []
    poly = lane_poly[0].astype(np.float32)
    danger = []
    for d in dets:
        if d.label not in VEHICLES:
            continue
        x1, y1, x2, y2 = d.box
        bottom_center = ((x1 + x2) / 2, float(y2))
        if cv2.pointPolygonTest(poly, bottom_center, False) >= 0 and (y2 - y1) / h > min_height_ratio:
            danger.append(d)
    return danger
