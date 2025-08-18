import cv2
import numpy as np
# Pré-processamento simples p/ OCR: recorta, amplia, filtra e binariza

def crop_with_padding(img, box, pad: int = 4):
    h, w = img.shape[:2]
    x1, y1, x2, y2, _ = box
    x1p = max(0, x1 - pad)
    y1p = max(0, y1 - pad)
    x2p = min(w - 1, x2 + pad)
    y2p = min(h - 1, y2 + pad)
    return img[y1p:y2p, x1p:x2p]
def enhance_for_ocr(crop, target_height: int = 64):
    if crop is None or crop.size == 0:
        return crop
    # Redimensiona preservando aspecto
    h, w = crop.shape[:2]
    scale = target_height / max(1, h)
    crop = cv2.resize(crop, (int(w * scale), int(h * scale)),
    interpolation=cv2.INTER_CUBIC)
    # Converte para tons de cinza
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    # Filtro bilateral preserva bordas
    gray = cv2.bilateralFilter(gray, 9, 75, 75)
    # Equalização adaptativa
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)
    # Binarização adaptativa
    bw = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                cv2.THRESH_BINARY, 31, 10)
    return bw