# validador_placa.py
import re

class ValidadorPlaca:
    """
    Valida e corrige o texto da placa reconhecida pelo OCR.
    O pós-processamento contextual aumenta a confiabilidade do sistema,
    corrigindo erros comuns e validando contra padrões conhecidos[cite: 201].
    """
    def __init__(self):
        # Padrão regex para placas Mercosul (AAA0A00)
        self.padrao_mercosul = re.compile(r'^[A-Z]{3}\d[A-Z]\d{2}$')
        
        # Mapeamento para correções comuns de OCR [cite: 207]
        self.mapa_correcao = {
            'O': '0',
            'I': '1',
            'G': '6',
            'S': '5',
            'B': '8',
        }

    def corrigir_e_validar(self, texto_placa):
        """
        Aplica correções e valida se o texto corresponde ao padrão Mercosul.

        Args:
            texto_placa: O texto bruto extraído pelo OCR.

        Returns:
            O texto da placa corrigido e validado, ou None se inválido.
        """
        if not texto_placa or len(texto_placa) != 7:
            return None
        
        # Aplica correções com base no mapa
        # Ex: "B0I1GAB" -> "80116AB"
        # Esta é uma lógica simplificada. Uma abordagem mais robusta
        # verificaria a posição do caractere (letra vs. número).
        
        # Correção baseada na posição (padrão LLLNLNN)
        corrigido = ""
        for i, char in enumerate(texto_placa):
            # Posições 0, 1, 2, 4 devem ser letras
            if i in [0, 1, 2, 4]:
                if char.isdigit():
                    # Troca dígito por letra parecida (ex: 0 por O)
                    for letra, digito in self.mapa_correcao.items():
                        if char == digito:
                            char = letra
                            break
            # Posições 3, 5, 6 devem ser números
            else:
                if char.isalpha():
                    # Troca letra por dígito parecido
                    char = self.mapa_correcao.get(char, char)
            corrigido += char
            
        # Validação final com regex
        if self.padrao_mercosul.match(corrigido):
            return corrigido
        
        return None