# pre_processador_otimizado.py
import cv2
import numpy as np

class PreProcessadorROI:
    """
    Classe para aplicar técnicas de Pré-processamento Digital de Imagens (PDI)
    na Região de Interesse (ROI) da placa.
    Essas operações de baixo nível (Low-Level Processing) são cruciais para
    melhorar a qualidade da imagem antes do OCR.
    """
    def processar(self, roi):
        """
        Aplica um pipeline de filtros e transformações na ROI da placa.

        Args:
            roi: A imagem (recorte) da placa detectada.

        Returns:
            A imagem da placa pré-processada e pronta para o OCR.
        """
        # 1. Redimensionar para um tamanho padrão
        roi = cv2.resize(roi, (400, 100), interpolation=cv2.INTER_CUBIC)

        # 2. Conversão para escala de cinza
        gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

        # 3. Suavização / Remoção de Ruído
        # O filtro Gaussiano é eficaz para remover ruídos de alta frequência.
        blurred_roi = cv2.GaussianBlur(gray_roi, (5, 5), 0)

        # 4. Realce de Contraste (Opcional, mas útil)
        # O CLAHE melhora o contraste local.
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        contrast_roi = clahe.apply(blurred_roi)

        # 5. Binarização com Thresholding OTSU
        # Otsu determina automaticamente o melhor limiar, melhorando a binarização
        # em condições de iluminação variadas.
        _, binary_roi = cv2.threshold(contrast_roi, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

        # 6. Operações Morfológicas para Refinar
        # Fechamento (dilatação seguida de erosão) para fechar lacunas nos caracteres
        # Abertura (erosão seguida de dilatação) para remover ruído
        kernel_fechamento = np.ones((3,3), np.uint8)
        morph_roi = cv2.morphologyEx(binary_roi, cv2.MORPH_CLOSE, kernel_fechamento)

        kernel_abertura = np.ones((2,2), np.uint8)
        morph_roi = cv2.morphologyEx(morph_roi, cv2.MORPH_OPEN, kernel_abertura)
        
        return morph_roi