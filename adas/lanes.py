"""Classic lane detection: color mask + Canny + Hough transform + temporal smoothing."""
from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class Line:
    """Lane line in image coords as x = k*y + b (stable for near-vertical lines)."""
    k: float
    b: float

    def x_at(self, y: float) -> float:
        return self.k * y + self.b


@dataclass
class LaneConfig:
    # region of interest, as fractions of width/height (trapezoid)
    roi_top_y: float = 0.62
    roi_top_left_x: float = 0.44
    roi_top_right_x: float = 0.56
    roi_bottom_margin_x: float = 0.05
    canny_low: int = 50
    canny_high: int = 150
    hough_threshold: int = 20
    hough_min_len: int = 20
    hough_max_gap: int = 150
    min_abs_slope: float = 0.4   # discard near-horizontal segments (dy/dx)
    smoothing: float = 0.8       # EMA weight of previous frame
    max_missed: int = 10         # frames to keep last line when nothing is detected


def color_mask(bgr: np.ndarray) -> np.ndarray:
    """Keep white and yellow pixels (HLS color space is robust to shadows)."""
    hls = cv2.cvtColor(bgr, cv2.COLOR_BGR2HLS)
    white = cv2.inRange(hls, (0, 190, 0), (180, 255, 255))
    yellow = cv2.inRange(hls, (15, 30, 100), (35, 220, 255))
    return cv2.bitwise_or(white, yellow)


class LaneDetector:
    def __init__(self, cfg: LaneConfig | None = None):
        self.cfg = cfg or LaneConfig()
        self.left: Line | None = None
        self.right: Line | None = None
        self._missed = {"left": 0, "right": 0}

    def roi_polygon(self, w: int, h: int) -> np.ndarray:
        c = self.cfg
        return np.array([[
            (int(w * c.roi_bottom_margin_x), h),
            (int(w * c.roi_top_left_x), int(h * c.roi_top_y)),
            (int(w * c.roi_top_right_x), int(h * c.roi_top_y)),
            (int(w * (1 - c.roi_bottom_margin_x)), h),
        ]], dtype=np.int32)

    def edges(self, bgr: np.ndarray) -> np.ndarray:
        h, w = bgr.shape[:2]
        mask = color_mask(bgr)
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        gray = cv2.bitwise_and(gray, gray, mask=mask)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blur, self.cfg.canny_low, self.cfg.canny_high)
        roi = np.zeros_like(edges)
        cv2.fillPoly(roi, self.roi_polygon(w, h), 255)
        return cv2.bitwise_and(edges, roi)

    def _fit(self, segments: list[tuple[int, int, int, int]]) -> Line | None:
        """Least-squares fit x = k*y + b, weighted by segment length."""
        if not segments:
            return None
        ys, xs, ws = [], [], []
        for x1, y1, x2, y2 in segments:
            length = np.hypot(x2 - x1, y2 - y1)
            ys += [y1, y2]
            xs += [x1, x2]
            ws += [length, length]
        k, b = np.polyfit(ys, xs, 1, w=np.sqrt(ws))
        return Line(float(k), float(b))

    def _smooth(self, side: str, new: Line | None) -> Line | None:
        old = getattr(self, side)
        if new is None:
            self._missed[side] += 1
            return old if self._missed[side] <= self.cfg.max_missed else None
        self._missed[side] = 0
        if old is None:
            return new
        a = self.cfg.smoothing
        return Line(a * old.k + (1 - a) * new.k, a * old.b + (1 - a) * new.b)

    def detect(self, bgr: np.ndarray) -> tuple[Line | None, Line | None]:
        h, w = bgr.shape[:2]
        segs = cv2.HoughLinesP(self.edges(bgr), 1, np.pi / 180, self.cfg.hough_threshold,
                               minLineLength=self.cfg.hough_min_len, maxLineGap=self.cfg.hough_max_gap)
        left, right = [], []
        # reshape: OpenCV 4 returns (N, 1, 4), OpenCV 5 returns (N, 4)
        for x1, y1, x2, y2 in (segs.reshape(-1, 4) if segs is not None else []):
            if x1 == x2:
                continue
            slope = (y2 - y1) / (x2 - x1)
            if abs(slope) < self.cfg.min_abs_slope:
                continue
            # in image coords y grows downward: left lane has negative slope
            if slope < 0 and max(x1, x2) < w * 0.55:
                left.append((x1, y1, x2, y2))
            elif slope > 0 and min(x1, x2) > w * 0.45:
                right.append((x1, y1, x2, y2))
        self.left = self._smooth("left", self._fit(left))
        self.right = self._smooth("right", self._fit(right))
        return self.left, self.right

    def lane_polygon(self, h: int) -> np.ndarray | None:
        """Ego-lane polygon between the two lines, from the bottom of the frame to the ROI top."""
        if self.left is None or self.right is None:
            return None
        y_top, y_bot = int(h * self.cfg.roi_top_y), h
        return np.array([[
            (int(self.left.x_at(y_bot)), y_bot), (int(self.left.x_at(y_top)), y_top),
            (int(self.right.x_at(y_top)), y_top), (int(self.right.x_at(y_bot)), y_bot),
        ]], dtype=np.int32)
