"""Run the ADAS perception pipeline on a video.

Example:
    python run.py --input data/solidWhiteRight.mp4 --output output/solidWhiteRight.mp4
"""
import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np

from adas.alerts import forward_collision, lane_departure, lane_offset
from adas.draw import draw_detections, draw_hud, draw_lanes
from adas.lanes import LaneDetector


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--model", default="yolo11n.pt", help="YOLO weights")
    ap.add_argument("--no-objects", action="store_true", help="lane detection only")
    ap.add_argument("--max-frames", type=int, default=0)
    args = ap.parse_args()

    cap = cv2.VideoCapture(args.input)
    if not cap.isOpened():
        raise SystemExit(f"cannot open {args.input}")
    fps_in = cap.get(cv2.CAP_PROP_FPS) or 25
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(args.output, cv2.VideoWriter_fourcc(*"mp4v"), fps_in, (w, h))

    lanes = LaneDetector()
    objects = None
    if not args.no_objects:
        from adas.objects import ObjectDetector
        objects = ObjectDetector(args.model)

    stats = {"frames": 0, "both_lanes_found": 0, "ldw_frames": 0, "fcw_frames": 0,
             "lane_ms": [], "object_ms": []}
    fps = 0.0
    while True:
        ok, frame = cap.read()
        if not ok or (args.max_frames and stats["frames"] >= args.max_frames):
            break
        t0 = time.perf_counter()
        left, right = lanes.detect(frame)
        t1 = time.perf_counter()
        dets = objects.detect(frame) if objects else []
        t2 = time.perf_counter()

        poly = lanes.lane_polygon(h)
        offset = lane_offset(left, right, w, h)
        ldw = lane_departure(offset)
        danger = forward_collision(dets, poly, h)

        fps = 0.9 * fps + 0.1 * (1 / max(t2 - t0, 1e-6)) if stats["frames"] else 1 / max(t2 - t0, 1e-6)
        draw_lanes(frame, left, right, poly, int(h * lanes.cfg.roi_top_y), ldw)
        draw_detections(frame, dets, danger)
        draw_hud(frame, offset, ldw, bool(danger), fps)
        writer.write(frame)

        stats["frames"] += 1
        stats["both_lanes_found"] += poly is not None
        stats["ldw_frames"] += ldw
        stats["fcw_frames"] += bool(danger)
        stats["lane_ms"].append((t1 - t0) * 1000)
        stats["object_ms"].append((t2 - t1) * 1000)

    cap.release()
    writer.release()

    n = max(stats["frames"], 1)
    summary = {
        "input": args.input,
        "frames": stats["frames"],
        "lane_detection_rate": round(stats["both_lanes_found"] / n, 3),
        "ldw_frames": stats["ldw_frames"],
        "fcw_frames": stats["fcw_frames"],
        "lane_ms_mean": round(float(np.mean(stats["lane_ms"])), 2),
        "object_ms_mean": round(float(np.mean(stats["object_ms"])), 2) if objects else None,
    }
    print(json.dumps(summary, indent=2))
    Path(args.output).with_suffix(".json").write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
