import re
from difflib import SequenceMatcher

class ValidadorPlaca:
    def __init__(self):
        # Padrões oficiais de placas brasileiras
        self.padrao_mercosul = re.compile(r'^[A-Z]{3}[0-9][A-Z][0-9]{2}$')  # LLLNLNN
        self.padrao_antigo = re.compile(r'^[A-Z]{3}[0-9]{4}$')               # LLLNNNN

        # Correções comuns e mais seguras (evita mapeamentos muito agressivos)
        self.map_num2let = {'0':'O', '1':'I', '2':'Z', '4':'A', '5':'S', '6':'G', '8':'B'}
        self.map_let2num = {'O':'0', 'Q':'0', 'D':'0', 'U':'0', 'I':'1', 'L':'1', 'Z':'2', 'S':'5', 'G':'6', 'B':'8'}

        # Histórico de consolidação
        self.historico_placas = {}
        self.contador_confianca = {}

    def corrigir_e_validar(self, texto_placa):
        if not texto_placa:
            return None

        texto = texto_placa.upper()

        # Garante tamanho de 7 caracteres
        if len(texto) < 7:
            return None
        if len(texto) > 7:
            texto = texto[:7]

        # Corrige posições especificamente conforme o padrão Mercosul (LLLNLNN)
        texto_corrigido = self._corrigir_por_posicao(texto)

        # Validação direta
        if self._valido(texto_corrigido):
            return self._consolidar(texto_corrigido)

        # Tenta uma única troca no 4º e 5º caracteres (pontos críticos do Mercosul)
        texto_try = self._tentar_troca_pontual(texto_corrigido)
        if self._valido(texto_try):
            return self._consolidar(texto_try)

        return None

    def _valido(self, s):
        return self.padrao_mercosul.match(s) is not None or self.padrao_antigo.match(s) is not None

    def _corrigir_por_posicao(self, texto):
        """Aplica correções com base na posição esperada (Mercosul: LLLNLNN)."""
        padrao_posicoes = [True, True, True, False, True, False, False]  # L=Letra, N=Número
        resultado = []
        for i, char in enumerate(texto):
            deve_ser_letra = padrao_posicoes[i]
            if deve_ser_letra and char.isdigit():
                char = self.map_num2let.get(char, char)
            elif not deve_ser_letra and char.isalpha():
                char = self.map_let2num.get(char, char)
            resultado.append(char)
        return "".join(resultado)

    def _tentar_troca_pontual(self, texto):
        """Tenta apenas mudanças mínimas nas posições 3 (N) e 4 (L) do padrão Mercosul."""
        chars = list(texto)
        # posição 3 (index 3) deve ser número
        if chars[3].isalpha():
            chars[3] = self.map_let2num.get(chars[3], chars[3])
        # posição 4 (index 4) deve ser letra
        if chars[4].isdigit():
            chars[4] = self.map_num2let.get(chars[4], chars[4])
        return "".join(chars)

    def _consolidar(self, placa):
        from time import time
        agora = time()
        # Remove histórico antigo (>30s)
        self.historico_placas = {p: t for p, t in self.historico_placas.items() if agora - t < 30}

        # Verifica similaridade com histórico (mais rígido para evitar juntar placas diferentes)
        for placa_existente in list(self.historico_placas.keys()):
            if SequenceMatcher(None, placa, placa_existente).ratio() > 0.9:
                self.contador_confianca[placa_existente] = self.contador_confianca.get(placa_existente, 0) + 1
                self.historico_placas[placa_existente] = agora
                return placa_existente

        # Nova placa
        self.historico_placas[placa] = agora
        self.contador_confianca[placa] = 1
        return placa

    def limpar_historico(self):
        self.historico_placas.clear()
        self.contador_confianca.clear()
