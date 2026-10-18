"""Road object detection with a pretrained YOLO model (COCO classes)."""
from dataclasses import dataclass

import numpy as np

# COCO classes relevant for driving
ROAD_CLASSES = {"person", "bicycle", "car", "motorcycle", "bus", "truck", "traffic light", "stop sign"}


@dataclass
class Detection:
    label: str
    conf: float
    box: tuple[int, int, int, int]  # x1, y1, x2, y2


class ObjectDetector:
    def __init__(self, model: str = "yolo11n.pt", conf: float = 0.35, imgsz: int = 640):
        from ultralytics import YOLO  # imported lazily so lane-only mode works without torch
        self.model = YOLO(model)
        self.conf = conf
        self.imgsz = imgsz
        names = self.model.names
        self.class_ids = [i for i, n in names.items() if n in ROAD_CLASSES]

    def detect(self, bgr: np.ndarray) -> list[Detection]:
        res = self.model.predict(bgr, conf=self.conf, imgsz=self.imgsz,
                                 classes=self.class_ids, verbose=False)[0]
        out = []
        for box, cls, conf in zip(res.boxes.xyxy.cpu().numpy(), res.boxes.cls.cpu().numpy(),
                                  res.boxes.conf.cpu().numpy()):
            out.append(Detection(self.model.names[int(cls)], float(conf), tuple(int(v) for v in box)))
        return out
