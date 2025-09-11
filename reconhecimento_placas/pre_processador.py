# pre_processador.py - VERSÃO SIMPLIFICADA
import cv2
import numpy as np

class PreProcessadorROI:
    def processar(self, roi):
        """
        Pré-processamento MÍNIMO para não destruir os caracteres
        """
        try:
            # 1. Redimensionar mantendo aspecto
            altura_alvo = 100
            proporcao = altura_alvo / roi.shape[0]
            largura_alvo = int(roi.shape[1] * proporcao)
            roi = cv2.resize(roi, (largura_alvo, altura_alvo), interpolation=cv2.INTER_CUBIC)

            # 2. Escala de cinza
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

            # 3. Apenas um leve blur para reduzir ruído
            blurred = cv2.GaussianBlur(gray, (3, 3), 0)

            # 4. Binarização NORMAL (sem operações morfológicas agressivas)
            _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

            return binary

        except Exception as e:
            print(f"Erro no pré-processamento: {e}")
            return roi  # Retorna original se der erro