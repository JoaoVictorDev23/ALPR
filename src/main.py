import argparse
import os
import sys
import cv2
from rich import print

from detector import PlateDetector
from ocr import PlateOCR
from preprocess import crop_with_padding, enhance_for_ocr
from viz import draw_box_and_label
from utils import ensure_dir, csv_logger, timestamp


def parse_args():
    p = argparse.ArgumentParser(description="ALPR – YOLO + PaddleOCR")
    p.add_argument("--source", type=str, default="0", help="0 (webcam) ou caminho de imagem/vídeo/pasta")
    p.add_argument("--model", type=str, default="./models/lp_detector.pt", help="pesos YOLO treinados p/ placas")
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--iou", type=float, default=0.45)
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--save-video", action="store_true")
    p.add_argument("--save-csv", action="store_true")
    p.add_argument("--output", type=str, default="./data/outputs")
    return p.parse_args()

def open_source(src: str):
    if src.isdigit():
        cap = cv2.VideoCapture(int(src))
        if not cap.isOpened():
            raise RuntimeError("Não foi possível abrir a webcam")
        return cap, True
    if os.path.isdir(src):
        # Lê imagens da pasta em ordem
        files = sorted([os.path.join(src, f) for f in os.listdir(src) if f.lower().endswith((".jpg",".png",".jpeg"))])
        return files, False
    if os.path.isfile(src):
        # decide se é imagem única
        if src.lower().endswith((".jpg",".png",".jpeg")):
            return [src], False
        else:
            cap = cv2.VideoCapture(src)
        if not cap.isOpened():
            raise RuntimeError("Não foi possível abrir o vídeo")
        return cap, True
    raise FileNotFoundError(f"Fonte não encontrada: {src}")

def main():
    args = parse_args()
    # Inicializa módulos
    if not os.path.exists(args.model):
        print(f"[red]Atenção:[/red] pesos não encontrados em {args.model}. Treine ou adicione um .pt de LPD.")
        sys.exit(1)
    detector = PlateDetector(args.model, device=args.device, conf=args.conf, iou=args.iou)
    ocr = PlateOCR(lang="en", use_angle_cls=True)
    ensure_dir(args.output)
    csv_f = writer = None
    if args.save_csv:
        csv_f, writer, csv_path = csv_logger(args.output)
        print(f"[green]CSV:[/green] {csv_path}")

    src, is_video_stream = open_source(args.source)

    vw = None
    if args.save_video and is_video_stream:
        # Configura VideoWriter assim que tivermos o primeiro frame
        pass

    def process_frame(frame, frame_idx: int):
        nonlocal vw
        result = detector.detect(frame)
        boxes = detector.extract_boxes(result)
        for box in boxes:
            crop = crop_with_padding(frame, box, pad=6)
            proc = enhance_for_ocr(crop)
            text, score = ocr.read_text(proc)
            label = text if text else "PLATE"
            draw_box_and_label(frame, box, label)
            if writer is not None:
                x1, y1, x2, y2, conf = box
                writer.writerow([frame_idx, x1, y1, x2, y2, f"{conf:.3f}",
                                 text or "", f"{score:.3f}"])
                if args.save_video and is_video_stream:
                    if vw is None:
                        h, w = frame.shape[:2]
                        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                        out_path = os.path.join(args.output, f"alpr-{timestamp()}.mp4")
                        vw = cv2.VideoWriter(out_path, fourcc, 30, (w, h))
                        print(f"[green]Vídeo:[/green] {out_path}")
                    vw.write(frame)
                return frame
            if is_video_stream:
                cap = src
            idx = 0
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                idx += 1
                frame = process_frame(frame, idx)
                cv2.imshow("ALPR – YOLO + OCR", frame)
                if cv2.waitKey(1) & 0xFF == 27:  # ESC
                    break
            cap.release()
        else:
        #Lista de imagens
            for i, path in enumerate(src, start=1):
                img = cv2.imread(path)
                if img is None:
                    print(f"[yellow]Aviso:[/yellow] não foi possível abrir {path}")
                    continue
                out = process_frame(img, i)
                cv2.imshow("ALPR – YOLO + OCR", out)
                cv2.waitKey(0)
    cv2.destroyAllWindows()

    if csv_f is not None:
        csv_f.close()
if __name__ == "__main__":
    main()


