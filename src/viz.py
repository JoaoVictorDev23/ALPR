import cv2
from typing import Tuple

FONT = cv2.FONT_HERSHEY_SIMPLEX

def draw_box_and_label(img, box, label: str, color: Tuple[int, int, int] =
(0, 255, 0)):
    x1, y1, x2, y2, conf = box
    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
    txt = f"{label} ({conf:.2f})" if label else f"{conf:.2f}"
    (tw, th), _ = cv2.getTextSize(txt, FONT, 0.6, 2)
    y = max(20, y1 - 10)
    cv2.rectangle(img, (x1, y - th - 6), (x1 + tw + 6, y), color, -1)
    cv2.putText(img, txt, (x1 + 3, y - 4), FONT, 0.6, (0, 0, 0), 2,
cv2.LINE_AA)