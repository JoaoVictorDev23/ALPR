import cv2
import numpy as np
import pytesseract
import re
import os

class LeitorOCR:
    def __init__(self, tesseract_cmd=r"C:\\Program Files\\Tesseract-OCR\\tesseract.exe",
                 tessdata_dir=r"C:\\Program Files\\Tesseract-OCR\\tessdata",
                 langs=("eng", "por")):
        if tesseract_cmd:
            pytesseract.pytesseract.tesseract_cmd = tesseract_cmd
        if tessdata_dir:
            os.environ['TESSDATA_PREFIX'] = tessdata_dir

        self.lang = "+".join(langs)

    def _limpar_texto_placa(self, texto):
        texto = (texto or "").upper()
        return re.sub(r"[^A-Z0-9]", "", texto)

    def ler_placa(self, imagem_processada):
        if imagem_processada is None:
            return ""

        img = imagem_processada
        if len(img.shape) == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        base_cfg = "--oem 1 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 -c user_defined_dpi=300"

        # 1ª tentativa: PSM 7 (linha única)
        try:
            raw = pytesseract.image_to_string(img, lang=self.lang, config=f"{base_cfg} --psm 7")
            texto = self._limpar_texto_placa(raw)
            if 6 <= len(texto) <= 7:
                return texto[:7]
        except Exception:
            pass

        # fallback: PSM 11 (texto esparso)
        try:
            raw = pytesseract.image_to_string(img, lang=self.lang, config=f"{base_cfg} --psm 11")
            texto = self._limpar_texto_placa(raw)
            if texto:
                return texto[:7]
        except Exception:
            pass

        return ""
