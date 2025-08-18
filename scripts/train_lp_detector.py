"""
Treinamento do detector de placas com Ultralytics YOLO.
1) Prepare um dataset no formato YOLO:
 - images/train, images/val, images/test
 - labels/train, labels/val, labels/test (arquivos .txt com [class x_center
8
y_center width height])
 - Em 'configs/dataset.yaml' aponte caminhos e classes
2) Execute:
 python -m scripts.train_lp_detector \
 --model yolo11n.pt \
 --data ./configs/dataset.yaml \
 --epochs 100 --imgsz 640 --batch 16 --device cuda:0
Resultado: melhor checkpoint em runs/detect/train/weights/best.pt
"""

import argparse
from ultralytics import YOLO

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--model', type=str, default='yolo11n.pt', help='backbone base (ou yolov8n.pt)')
    p.add_argument('--data', type=str, default='./configs/dataset.yaml')
    p.add_argument('--epochs', type=int, default=100)
    p.add_argument('--imgsz', type=int, default=640)
    p.add_argument('--batch', type=int, default=16)
    p.add_argument('--device', type=str, default='cpu')
    return p.parse_args()

def main():
    args = parse_args()
    model = YOLO(args.model)
    model.train(data=args.data,
                epochs=args.epochs,
                imgsz=args.imgsz,
                batch=args.batch,
                device=args.device)

if __name__ == '__main__':
    main()

