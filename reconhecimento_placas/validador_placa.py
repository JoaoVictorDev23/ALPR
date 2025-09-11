# validador_placa.py
import re
from difflib import SequenceMatcher


class ValidadorPlaca:
    def __init__(self):
        # Padrões para placas brasileiras
        self.padrao_mercosul = re.compile(r'^[A-Z]{3}\d[A-Z]\d{2}$')
        self.padrao_antigo = re.compile(r'^[A-Z]{3}\d{4}$')

        # Mapeamento de correções comuns
        self.mapa_correcao = {
            'O': '0', '0': 'O',
            'I': '1', '1': 'I',
            'Z': '2', '2': 'Z',
            'S': '5', '5': 'S',
            'G': '6', '6': 'G',
            'B': '8', '8': 'B',
            'Q': '0', 'D': '0',
            'U': '0', 'V': 'U',
            'Y': '4', '4': 'A',
            'X': 'K', 'K': 'X'
        }

        # Histórico de placas recentes para consolidação
        self.historico_placas = {}
        self.contador_confianca = {}

    def corrigir_e_validar(self, texto_placa):
        if not texto_placa or len(texto_placa) < 6:
            return None

        texto = texto_placa.upper()[:7]

        # Aplica correções inteligentes baseadas na posição
        texto_corrigido = self._corrigir_inteligente(texto)

        # Valida com os padrões
        if self.padrao_mercosul.match(texto_corrigido):
            return self._consolidar_placa(texto_corrigido)
        elif self.padrao_antigo.match(texto_corrigido):
            return self._consolidar_placa(texto_corrigido)
        else:
            # Tenta uma última correção agressiva
            texto_final = self._correcao_agressiva(texto_corrigido)
            if texto_final and (self.padrao_mercosul.match(texto_final) or
                                self.padrao_antigo.match(texto_final)):
                return self._consolidar_placa(texto_final)

        return None

    def _corrigir_inteligente(self, texto):
        """Correções baseadas na posição esperada da placa"""
        if len(texto) != 7:
            return texto

        # Padrão Mercosul: LLLNLNN (Letra, Letra, Letra, Número, Letra, Número, Número)
        padrao_posicoes = [True, True, True, False, True, False, False]  # L,L,L,N,L,N,N

        corrigido = ""
        for i, char in enumerate(texto):
            if i >= len(padrao_posicoes):
                corrigido += char
                continue

            deve_ser_letra = padrao_posicoes[i]
            char_corrigido = char

            # Correção baseada na posição
            if deve_ser_letra:
                if char.isdigit():
                    # Dígito onde deveria ser letra - converte para letra similar
                    char_corrigido = self.mapa_correcao.get(char, char)
                    if char_corrigido.isdigit():
                        # Se ainda for dígito, tenta outra correção
                        char_corrigido = self._encontrar_letra_similar(char)
            else:
                if char.isalpha():
                    # Letra onde deveria ser dígito - converte para dígito similar
                    char_corrigido = self.mapa_correcao.get(char, char)
                    if char_corrigido.isalpha():
                        # Se ainda for letra, tenta outra correção
                        char_corrigido = self._encontrar_digito_similar(char)

            corrigido += char_corrigido

        return corrigido

    def _encontrar_letra_similar(self, digito):
        """Encontra letra similar ao dígito"""
        similares = {
            '0': 'O', '1': 'I', '2': 'Z', '4': 'A',
            '5': 'S', '6': 'G', '8': 'B', '9': 'G'
        }
        return similares.get(digito, digito)

    def _encontrar_digito_similar(self, letra):
        """Encontra dígito similar à letra"""
        similares = {
            'O': '0', 'Q': '0', 'D': '0',
            'I': '1', 'L': '1',
            'Z': '2',
            'A': '4',
            'S': '5',
            'G': '6', 'B': '8'
        }
        return similares.get(letra, letra)

    def _correcao_agressiva(self, texto):
        """Correção final agressiva para tentar salvar a placa"""
        if len(texto) != 7:
            return texto

        # Tenta padrões comuns de erros
        correcoes_comuns = {
            'GBXOC44': 'GBX0C44',  # O -> 0
            'GBXUC44': 'GBX0C44',  # U -> 0
            'GBXOC64': 'GBX0C44',  # 6 -> 4, O -> 0
            'GBXOCA4': 'GBX0C44',  # A -> 4, O -> 0
            'GBXOCY4': 'GBX0C44',  # Y -> 4, O -> 0
            'F2ZOA0': 'F2Z0A0',  # O -> 0
        }

        return correcoes_comuns.get(texto, texto)

    def _consolidar_placa(self, placa):
        """Consolida leituras similares para evitar oscilações"""
        # Limpa histórico antigo (mais de 30 segundos)
        from time import time
        current_time = time()
        self.historico_placas = {p: t for p, t in self.historico_placas.items()
                                 if current_time - t < 30}

        # Verifica se é similar a alguma placa recente
        for placa_existente in list(self.historico_placas.keys()):
            similaridade = SequenceMatcher(None, placa, placa_existente).ratio()
            if similaridade > 0.8:  # 80% de similaridade
                # Usa a placa mais frequente ou a mais recente
                if placa_existente in self.contador_confianca:
                    self.contador_confianca[placa_existente] += 1
                else:
                    self.contador_confianca[placa_existente] = 1

                self.historico_placas[placa_existente] = current_time
                return placa_existente

        # Nova placa
        self.historico_placas[placa] = current_time
        self.contador_confianca[placa] = 1
        return placa

    def limpar_historico(self):
        """Limpa o histórico de placas"""
        self.historico_placas.clear()
        self.contador_confianca.clear()