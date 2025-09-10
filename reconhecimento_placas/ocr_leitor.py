# ocr_leitor.py
import pytesseract
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
class LeitorOCR:
    """
    Classe responsável pelo Reconhecimento Óptico de Caracteres (OCR).
    Utiliza o Tesseract, que pode ser configurado para usar Redes Neurais de
    Longa Duração (LSTM) para melhorar a precisão na leitura de sequências
    de caracteres[cite: 195, 197].
    """
    def __init__(self, tesseract_path=None):
        """
        Configura o caminho para o executável do Tesseract, se necessário.
        """
        if tesseract_path:
            pytesseract.pytesseract.tesseract_cmd = tesseract_path

    # ocr_leitor.py
# ...
class LeitorOCR:
    # ...
    def ler_placa(self, imagem_processada):
        # Aumentamos o filtro de caracteres para incluir 'o' e 'O'
        # Isso ajuda o Tesseract a tentar reconhecer letras em posições de números
        config = '--psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
        
        try:
            texto = pytesseract.image_to_string(imagem_processada, config=config)
            return "".join(texto.split()).upper()
        except Exception as e:
            print(f"[ERRO OCR] Falha ao ler a imagem: {e}")
            return ""