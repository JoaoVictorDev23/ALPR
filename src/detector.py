from ultralytics import YOLO
from typing import List, Tuple
import numpy as np

class PlateDetector:
    def __init__(self, weights_path: str, device: str = "cpu", conf: float =
0.25, iou: float = 0.45):
        self.model = YOLO(weights_path)
        self.model.to(device)
        self.device = device
        self.conf = conf
        self.iou = iou
    def detect(self, image: np.ndarray):
        # Ultralytics aceita BGR (cv2) diretamente
        results = self.model.predict(source=image, conf=self.conf,
        iou=self.iou, verbose=False)
        return results[0]

    @staticmethod
    def extract_boxes(result) -> List[Tuple[int, int, int, int, float]]:
        boxes = []
        if result and result.boxes is not None:
            for b in result.boxes:
                x1, y1, x2, y2 = b.xyxy[0].tolist()
                conf = float(b.conf[0].item()) if b.conf is not None else 0.0
                boxes.append((int(x1), int(y1), int(x2), int(y2), conf))
        return boxes