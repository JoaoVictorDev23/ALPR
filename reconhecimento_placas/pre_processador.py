# pre_processador.py
import cv2
import numpy as np

class PreProcessadorROI:
    """
    Classe para aplicar técnicas de Pré-processamento Digital de Imagens (PDI)
    na Região de Interesse (ROI) da placa.
    Essas operações de baixo nível (Low-Level Processing) são cruciais para
    melhorar a qualidade da imagem antes do OCR[cite: 82].
    """
    def processar(self, roi):
        """
        Aplica um pipeline de filtros e transformações na ROI da placa.

        Args:
            roi: A imagem (recorte) da placa detectada.

        Returns:
            A imagem da placa pré-processada e pronta para o OCR.
        """
        # Redimensionar para um tamanho padrão pode melhorar a consistência do OCR
        roi = cv2.resize(roi, (300, 100), interpolation=cv2.INTER_CUBIC)

        # 1. Conversão para escala de cinza
        # A cor pode ser descartada para focar na forma dos caracteres.
        gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

        # 2. Suavização / Remoção de Ruído
        # O filtro Gaussiano é eficaz para remover ruídos de alta frequência[cite: 83].
        blurred_roi = cv2.GaussianBlur(gray_roi, (5, 5), 0)

        # 3. Realce de Contraste
        # A equalização de histograma adaptativa (CLAHE) melhora o contraste local.
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        contrast_roi = clahe.apply(blurred_roi)

        # 4. Binarização
        # O Thresholding Adaptativo é robusto a diferentes condições de iluminação[cite: 83].
        binary_roi = cv2.adaptiveThreshold(
            contrast_roi, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 11, 2
        )

        # 5. Operações Morfológicas (Opcional, mas útil)
        # A abertura (erosão seguida de dilatação) pode remover pequenos ruídos (sal).
        kernel = np.ones((2, 2), np.uint8)
        morph_roi = cv2.morphologyEx(binary_roi, cv2.MORPH_OPEN, kernel)
        
        return morph_roi