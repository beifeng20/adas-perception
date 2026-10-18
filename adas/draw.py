"""Visualization of the perception results."""
import cv2
import numpy as np

from .lanes import Line
from .objects import Detection

GREEN, RED, YELLOW, WHITE = (0, 200, 0), (0, 0, 255), (0, 220, 255), (255, 255, 255)


def draw_lanes(img: np.ndarray, left: Line | None, right: Line | None, poly: np.ndarray | None,
               y_top: int, warn: bool) -> None:
    h = img.shape[0]
    if poly is not None:
        overlay = img.copy()
        cv2.fillPoly(overlay, poly, RED if warn else GREEN)
        cv2.addWeighted(overlay, 0.25, img, 0.75, 0, dst=img)
    for line in (left, right):
        if line is not None:
            cv2.line(img, (int(line.x_at(h)), h), (int(line.x_at(y_top)), y_top), YELLOW, 6)


def draw_detections(img: np.ndarray, dets: list[Detection], danger: list[Detection]) -> None:
    for d in dets:
        color = RED if d in danger else (255, 140, 0)
        x1, y1, x2, y2 = d.box
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        cv2.putText(img, f"{d.label} {d.conf:.2f}", (x1, max(y1 - 6, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)


def draw_hud(img: np.ndarray, offset: float | None, ldw: bool, fcw: bool, fps: float) -> None:
    cv2.rectangle(img, (0, 0), (360, 92), (0, 0, 0), -1)
    off = "n/a" if offset is None else f"{offset:+.2f} lane widths"
    cv2.putText(img, f"Offset: {off}", (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, WHITE, 2)
    cv2.putText(img, f"FPS: {fps:.1f}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, WHITE, 2)
    msgs = (["LANE DEPARTURE"] if ldw else []) + (["COLLISION WARNING"] if fcw else [])
    if msgs:
        cv2.putText(img, " | ".join(msgs), (10, 78), cv2.FONT_HERSHEY_SIMPLEX, 0.6, RED, 2)
