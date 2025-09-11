# validador_placa.py - VERSÃO DEBUG
import re


class ValidadorPlaca:
    def __init__(self):
        self.padrao_mercosul = re.compile(r'^[A-Z]{3}\d[A-Z]\d{2}$')
        self.padrao_antigo = re.compile(r'^[A-Z]{3}\d{4}$')

    def corrigir_e_validar(self, texto_placa):
        if not texto_placa or len(texto_placa) < 6:
            print(f"Texto muito curto: '{texto_placa}'")
            return None

        texto = texto_placa.upper()[:7]
        print(f"Validando: '{texto}'")  # DEBUG

        # Tenta padrão Mercosul
        if self.padrao_mercosul.match(texto):
            print(f"Placa Mercosul válida: {texto}")
            return texto

        # Tenta padrão antigo
        if self.padrao_antigo.match(texto):
            print(f"Placa antiga válida: {texto}")
            return texto

        # PARA DEBUG: Aceita QUALQUER texto com 7 caracteres
        if len(texto) == 7:
            print(f"DEBUG - Aceitando placa não validada: {texto}")
            return texto

        print(f"Placa inválida: {texto}")
        return None