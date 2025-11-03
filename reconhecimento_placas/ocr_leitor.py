import cv2
import numpy as np
import pytesseract
import re
import os

class LeitorOCR:
    """OCR da ROI de placa via Tesseract (eng+por), com fallback de PSM."""

    def __init__(self, tesseract_cmd=r"C:\\Program Files\\Tesseract-OCR\\tesseract.exe",
                 tessdata_dir=r"C:\\Program Files\\Tesseract-OCR\\tessdata",
                 langs=("eng", "por")):
        # Caminho do binário (Windows). Em Linux/Mac costuma ser desnecessário.
        if tesseract_cmd:
            pytesseract.pytesseract.tesseract_cmd = tesseract_cmd
        # Diretório dos modelos (traineddata)
        if tessdata_dir:
            os.environ['TESSDATA_PREFIX'] = tessdata_dir

        # Idiomas combinados para o engine (ex.: "eng+por")
        self.lang = "+".join(langs)

    def _limpar_texto_placa(self, texto):
        """Normaliza saída do OCR: maiúsculas e somente A–Z/0–9."""
        texto = (texto or "").upper()
        return re.sub(r"[^A-Z0-9]", "", texto)

    def ler_placa(self, imagem_processada):
        """
        Executa OCR na ROI já pré-processada.
        Estratégia:
          1) PSM 7  → assume uma linha única (melhor quando a placa está alinhada).
          2) PSM 11 → texto esparso (mais tolerante a desalinhamento/ruído).
        Retorna até 7 caracteres (ou "" se falhar).
        """
        if imagem_processada is None:
            return ""

        img = imagem_processada
        # Trabalhar em escala de cinza tende a ser mais robusto p/ OCR
        if len(img.shape) == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Base de configuração:
        # --oem 1: engine LSTM moderna
        # whitelist: restringe alfabeto ao padrão de placas (A–Z / 0–9)
        # DPI lógico: ajuda na calibração de tamanho de fonte
        base_cfg = "--oem 1 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 -c user_defined_dpi=300"

        # ---- 1ª tentativa: PSM 7 (Single line) ----
        # Útil quando a ROI da placa está bem alinhada/retangular.
        try:
            raw = pytesseract.image_to_string(img, lang=self.lang, config=f"{base_cfg} --psm 7")
            texto = self._limpar_texto_placa(raw)
            # Placas BR geralmente têm 7 caracteres; aceitamos 6–7 por segurança.
            if 6 <= len(texto) <= 7:
                return texto[:7]
        except Exception:
            # Erros do Tesseract não devem quebrar o pipeline
            pass

        # ---- Fallback: PSM 11 (Sparse text) ----
        # Mais permissivo: recupera quando há desalinhamento/ruído/corte.
        try:
            raw = pytesseract.image_to_string(img, lang=self.lang, config=f"{base_cfg} --psm 11")
            texto = self._limpar_texto_placa(raw)
            if texto:
                return texto[:7]
        except Exception:
            pass

        # Sem leitura confiável
        return ""
