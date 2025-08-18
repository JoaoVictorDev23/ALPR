import csv
import os
from datetime import datetime

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def timestamp() -> str:
    return datetime.now().strftime("%Y%m%d-%H%M%S")

def csv_logger(out_dir: str, name_prefix: str="detections"):
    ensure_dir(out_dir)
    path = os.path.join(out_dir, f"{name_prefix}-{timestamp()}.csv")
    f = open(path, mode="w", newline="", encoding="utf-8")
    writer = csv.writer(f)
    writer.writerow(["frame", "x1", "y1", "x2", "y2", "det_conf", "plate_text", "ocr_score"])
    return f, writer, path