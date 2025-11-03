# pre_processador.py - VERSÃO APRIMORADA
import cv2
import numpy as np

class PreProcessadorROI:
    def __init__(self, metodo: str = "robusto"):
        """
        metodo:
          - "robusto" (default): normaliza contraste, reduz ruído, binariza e faz morfologia,
                                 adiciona borda e redimensiona para altura ~110 px
        """
        self.metodo = metodo

    def _padronizar_tamanho(self, gray, altura_alvo=110):
        h, w = gray.shape[:2]
        if h == 0 or w == 0:
            return gray
        escala = altura_alvo / float(h)
        novo_w = max(1, int(w * escala))
        return cv2.resize(gray, (novo_w, altura_alvo), interpolation=cv2.INTER_CUBIC)

    def _auto_binarizar(self, gray):
        """Combina Otsu e adaptativo; escolhe o com maior separação (variância intra classes menor)."""
        # Otsu
        blur = cv2.GaussianBlur(gray, (3, 3), 0)
        _, bin_otsu = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Adaptativo
        bin_adap = cv2.adaptiveThreshold(
            gray, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 15, 5
        )

        # Heurística simples: escolhe o que tiver proporção de pixels pretos mais próxima de ~0.35–0.6
        def score(img):
            preto = 1.0 - (img.mean() / 255.0)
            return -abs(preto - 0.45)
        return bin_otsu if score(bin_otsu) >= score(bin_adap) else bin_adap

    def processar(self, roi):
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

        if self.metodo == "robusto":
            # 1) equalização local para lidar com sombras/alto brilho
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            gray = clahe.apply(gray)

            # 2) redução de ruído preservando bordas (melhora contornos dos caracteres)
            gray = cv2.bilateralFilter(gray, d=7, sigmaColor=60, sigmaSpace=60)

            # 3) padroniza tamanho para dar mais "dpi" ao Tesseract
            gray = self._padronizar_tamanho(gray, altura_alvo=110)

            # 4) binarização robusta
            binary = self._auto_binarizar(gray)

            # 5) morfologias leves: fechar para unir strokes; abrir para remover pontinhos
            kernel3 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            kernel2 = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
            binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel3, iterations=1)
            binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel2, iterations=1)

            # 6) adiciona borda branca para o Tesseract não cortar letras encostadas
            binary = cv2.copyMakeBorder(binary, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=255)

        elif self.metodo == "clahe_otsu":
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            gray = clahe.apply(gray)
            _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        elif self.metodo == "adaptive":
            binary = cv2.adaptiveThreshold(
                gray, 255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY, 11, 2
            )

        elif self.metodo == "sharpen_otsu":
            kernel = np.array([[0,-1,0], [-1,5,-1], [0,-1,0]])
            sharpened = cv2.filter2D(gray, -1, kernel)
            _, binary = cv2.threshold(sharpened, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        else:
            # fallback para manter compatibilidade
            _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        return binary
