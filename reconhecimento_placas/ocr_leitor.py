# ocr_leitor.py - VERSÃO DEFINITIVA COM EASYOCR
import easyocr
import cv2
import numpy as np
import re


class LeitorOCR:
    def __init__(self):
        """
        Inicializa o EasyOCR uma única vez para melhor performance.
        Usa os modelos para português e inglês.
        """
        print("Inicializando EasyOCR...")
        try:
            # Inicializa o reader com GPU False para melhor compatibilidade
            self.reader = easyocr.Reader(['pt', 'en'], gpu=False)
            print("EasyOCR inicializado com sucesso!")
        except Exception as e:
            print(f"Erro ao inicializar EasyOCR: {e}")
            self.reader = None

    def ler_placa(self, imagem_processada):
        """
        Lê o texto da placa processada usando EasyOCR.

        Args:
            imagem_processada: Imagem binária ou em escala de cinza da placa

        Returns:
            Texto da placa limpo e em maiúsculas, ou string vazia se falhar
        """
        if self.reader is None:
            print("EasyOCR não inicializado")
            return ""

        try:
            # Converte para RGB se necessário (EasyOCR espera RGB)
            if len(imagem_processada.shape) == 2:  # Imagem em escala de cinza
                imagem_rgb = cv2.cvtColor(imagem_processada, cv2.COLOR_GRAY2RGB)
            else:
                imagem_rgb = imagem_processada

            # Usa o EasyOCR para detectar texto
            resultados = self.reader.readtext(imagem_rgb, detail=0)

            if resultados:
                print(f"Resultados brutos do OCR: {resultados}")

                # Pega todos os resultados e junta (pode haver múltiplas detecções)
                texto_completo = " ".join(resultados)
                texto_limpo = self._limpar_texto_placa(texto_completo)

                print(f"Texto limpo: '{texto_limpo}'")
                return texto_limpo
            else:
                print("Nenhum texto detectado pelo OCR")
                return ""

        except Exception as e:
            print(f"Erro durante leitura OCR: {e}")
            return ""

    def _limpar_texto_placa(self, texto):
        """
        Limpa e formata o texto para padrão de placa veicular.

        Args:
            texto: Texto bruto do OCR

        Returns:
            Texto limpo e formatado
        """
        # Converte para maiúsculas e remove espaços
        texto = texto.upper().replace(" ", "")

        # Remove caracteres inválidos (mantém apenas letras e números)
        texto = re.sub(r'[^A-Z0-9]', '', texto)

        # Filtros específicos para placas brasileiras
        if len(texto) >= 6:
            # Prioriza textos com 7 caracteres (padrão Mercosul)
            if len(texto) == 7:
                return texto

            # Se tem mais de 7 caracteres, pega os primeiros 7
            elif len(texto) > 7:
                return texto[:7]

            # Se tem 6 caracteres, pode ser placa antiga
            else:
                return texto

        return ""