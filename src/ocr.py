from paddleocr import PaddleOCR
import numpy as np
from typing import Optional, Tuple

class PlateOCR:
    def __init__(self, lang: str = "en", use_angle_cls: bool = True, det:
bool = True):
        # PaddleOCR baixa pesos automaticamente na 1ª execução
        self.ocr = PaddleOCR(use_angle_cls=use_angle_cls, lang=lang, det=det)
    def read_text(self, image: np.ndarray) -> Tuple[Optional[str], float]:
        if image is None or image.size == 0:
            return None, 0.0
        result = self.ocr.ocr(image, cls=True)
        # Retorna melhor linha por score médio
        best_text, best_score = None, 0.0
        if result:
            lines = result[0]
            for line in lines:
                text = line[1][0]
                score = float(line[1][1])
                if score > best_score:
                    best_text, best_score = text, score
        return best_text, best_score
