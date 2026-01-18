 

import cv2
import numpy as np
import pytesseract
import os
import re
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict, Any
from enum import Enum, auto


class FaseCompilacao(Enum):
    
    ENTRADA = auto()           # Recepção da imagem fonte
    PRE_PROCESSAMENTO = auto() # Normalização do código-fonte visual
    ANALISE_LEXICA = auto()    # Segmentação em lexemas (caracteres)
    ANALISE_SINTATICA = auto() # Reconhecimento e classificação de tokens
    ANALISE_SEMANTICA = auto() # Validação de padrões e correção
    GERACAO_CODIGO = auto()    # Síntese da saída final


@dataclass
class Token:
    
    caractere: str
    posicao: int
    bounding_box: Tuple[int, int, int, int]
    confianca: float
    tipo: str = "DESCONHECIDO"
    imagem_segmento: Optional[np.ndarray] = None
    
    def __post_init__(self):
        """Classificação automática do tipo de token."""
        if self.caractere.isalpha():
            self.tipo = "LETRA"
        elif self.caractere.isdigit():
            self.tipo = "DIGITO"


@dataclass
class TabelaSimbolos:
    
    tokens: List[Token] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def adicionar_token(self, token: Token) -> None:
        
        self.tokens.append(token)
    
    def obter_lexema_completo(self) -> str:
        
        return "".join(t.caractere for t in self.tokens)
    
    def obter_confianca_media(self) -> float:
        
        if not self.tokens:
            return 0.0
        return sum(t.confianca for t in self.tokens) / len(self.tokens)
    
    def filtrar_por_tipo(self, tipo: str) -> List[Token]:
        
        return [t for t in self.tokens if t.tipo == tipo]


@dataclass
class ResultadoCompilacao:
    
    sucesso: bool
    texto_final: str
    tabela_simbolos: TabelaSimbolos
    fases_executadas: List[FaseCompilacao] = field(default_factory=list)
    erros: List[str] = field(default_factory=list)
    imagens_intermediarias: Dict[str, np.ndarray] = field(default_factory=dict)
    tempo_execucao_ms: float = 0.0


class OCRCompilador:
    
    
    # =========================================================================
    # CONSTANTES DE CONFIGURAÇÃO
    # =========================================================================
    
    # Padrões de placas brasileiras (expressões regulares)
    PADRAO_MERCOSUL = re.compile(r'^[A-Z]{3}[0-9][A-Z][0-9]{2}$')  # ABC1D23
    PADRAO_ANTIGO = re.compile(r'^[A-Z]{3}[0-9]{4}$')              # ABC1234
    
    # Mapeamentos para correção de erros comuns de OCR
    # Fundamentação: confusões típicas entre glifos visualmente similares
    MAPA_DIGITO_PARA_LETRA = {
        '0': 'O', '1': 'I', '2': 'Z', '4': 'A', 
        '5': 'S', '6': 'G', '8': 'B'
    }
    MAPA_LETRA_PARA_DIGITO = {
        'O': '0', 'Q': '0', 'D': '0', 'U': '0',
        'I': '1', 'L': '1', 'Z': '2', 'S': '5', 
        'G': '6', 'B': '8'
    }
    
    # Caracteres válidos para placas veiculares
    ALFABETO_VALIDO = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
    
    def __init__(
        self,
        tesseract_cmd: str = r"C:\Program Files\Tesseract-OCR\tesseract.exe",
        tessdata_dir: str = r"C:\Program Files\Tesseract-OCR\tessdata",
        idiomas: Tuple[str, ...] = ("eng", "por"),
        modo_debug: bool = False
    ):
        
        # Configuração do engine Tesseract
        if tesseract_cmd and os.path.exists(tesseract_cmd):
            pytesseract.pytesseract.tesseract_cmd = tesseract_cmd
        if tessdata_dir and os.path.exists(tessdata_dir):
            os.environ['TESSDATA_PREFIX'] = tessdata_dir
        
        self.idiomas = "+".join(idiomas)
        self.modo_debug = modo_debug
        
        # Configuração do Tesseract para reconhecimento de caracteres individuais
        # --oem 1: Utiliza engine LSTM (redes neurais recorrentes)
        # --psm 10: Trata a imagem como um único caractere
        self._config_char = (
            "--oem 1 --psm 10 "
            f"-c tessedit_char_whitelist={self.ALFABETO_VALIDO}"
        )
        
        # Configuração para linha única (fallback)
        self._config_linha = (
            "--oem 1 --psm 7 "
            f"-c tessedit_char_whitelist={self.ALFABETO_VALIDO} "
            "-c user_defined_dpi=300"
        )
    
    # =========================================================================
    # FASE 1: ENTRADA E PRÉ-PROCESSAMENTO
    # =========================================================================
    
    def _fase_pre_processamento(
        self, 
        imagem: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        
        # Conversão para escala de cinza
        if len(imagem.shape) == 3:
            cinza = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)
        else:
            cinza = imagem.copy()
        
        # Equalização adaptativa de histograma (CLAHE)
        # Melhora contraste local, essencial para placas com iluminação irregular
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        equalizado = clahe.apply(cinza)
        
        # Filtragem bilateral: suaviza ruído preservando bordas dos caracteres
        filtrado = cv2.bilateralFilter(equalizado, d=7, sigmaColor=50, sigmaSpace=50)
        
        # Padronização de tamanho (altura fixa para consistência de escala)
        altura_alvo = 100
        h, w = filtrado.shape[:2]
        if h > 0:
            escala = altura_alvo / float(h)
            novo_w = max(1, int(w * escala))
            filtrado = cv2.resize(
                filtrado, (novo_w, altura_alvo), 
                interpolation=cv2.INTER_CUBIC
            )
        
        # Binarização via método de Otsu (threshold automático)
        blur = cv2.GaussianBlur(filtrado, (3, 3), 0)
        _, binario = cv2.threshold(
            blur, 0, 255, 
            cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )
        
        # Operações morfológicas para limpeza
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        binario = cv2.morphologyEx(binario, cv2.MORPH_CLOSE, kernel)
        binario = cv2.morphologyEx(binario, cv2.MORPH_OPEN, kernel)
        
        return filtrado, binario
    
    # =========================================================================
    # FASE 2: ANÁLISE LÉXICA (SEGMENTAÇÃO)
    # =========================================================================
    
    def _fase_analise_lexica(
        self, 
        imagem_binaria: np.ndarray
    ) -> List[Tuple[np.ndarray, Tuple[int, int, int, int]]]:
        
        # Inversão se necessário (contornos detectam objetos brancos em fundo preto)
        # Verifica se o fundo é predominantemente branco
        if np.mean(imagem_binaria) > 127:
            imagem_trabalho = cv2.bitwise_not(imagem_binaria)
        else:
            imagem_trabalho = imagem_binaria.copy()
        
        # Detecção de contornos externos
        contornos, _ = cv2.findContours(
            imagem_trabalho, 
            cv2.RETR_EXTERNAL, 
            cv2.CHAIN_APPROX_SIMPLE
        )
        
        # Parâmetros para filtragem de contornos válidos
        altura_img, largura_img = imagem_binaria.shape[:2]
        area_minima = (altura_img * largura_img) * 0.005  # 0.5% da imagem
        area_maxima = (altura_img * largura_img) * 0.25   # 25% da imagem
        altura_minima = altura_img * 0.3   # Pelo menos 30% da altura
        proporcao_maxima = 1.5             # Largura/Altura máxima
        
        caracteres_segmentados = []
        
        for contorno in contornos:
            x, y, w, h = cv2.boundingRect(contorno)
            area = w * h
            proporcao = w / h if h > 0 else 0
            
            # Filtragem por critérios geométricos
            if (area_minima < area < area_maxima and 
                h >= altura_minima and 
                proporcao < proporcao_maxima):
                
                # Extrai o recorte do caractere com margem
                margem = 3
                x1 = max(0, x - margem)
                y1 = max(0, y - margem)
                x2 = min(largura_img, x + w + margem)
                y2 = min(altura_img, y + h + margem)
                
                recorte = imagem_binaria[y1:y2, x1:x2]
                bbox = (x, y, w, h)
                
                caracteres_segmentados.append((recorte, bbox))
        
        # Ordenação da esquerda para a direita (ordem de leitura)
        caracteres_segmentados.sort(key=lambda item: item[1][0])
        
        return caracteres_segmentados
    
    # =========================================================================
    # FASE 3: ANÁLISE SINTÁTICA (RECONHECIMENTO)
    # =========================================================================
    
    def _fase_analise_sintatica(
        self, 
        caracteres_segmentados: List[Tuple[np.ndarray, Tuple[int, int, int, int]]],
        imagem_cinza: np.ndarray
    ) -> TabelaSimbolos:
        
        tabela = TabelaSimbolos()
        tabela.metadata['total_segmentos'] = len(caracteres_segmentados)
        
        for idx, (recorte, bbox) in enumerate(caracteres_segmentados):
            # Preparação do recorte para OCR
            recorte_preparado = self._preparar_recorte_para_ocr(recorte)
            
            # Tentativa primária: reconhecimento de caractere único
            try:
                # Obtém dados detalhados do Tesseract (inclui confiança)
                dados = pytesseract.image_to_data(
                    recorte_preparado,
                    lang=self.idiomas,
                    config=self._config_char,
                    output_type=pytesseract.Output.DICT
                )
                
                caractere, confianca = self._extrair_melhor_resultado(dados)
                
            except Exception:
                caractere = ""
                confianca = 0.0
            
            # Fallback: se não reconheceu, tenta com PSM 7 (linha)
            if not caractere:
                try:
                    texto_bruto = pytesseract.image_to_string(
                        recorte_preparado,
                        lang=self.idiomas,
                        config=self._config_char
                    ).strip().upper()
                    caractere = texto_bruto[0] if texto_bruto else ""
                    confianca = 0.5 if caractere else 0.0
                except Exception:
                    caractere = ""
                    confianca = 0.0
            
            # Sanitização: apenas caracteres válidos
            if caractere and caractere in self.ALFABETO_VALIDO:
                token = Token(
                    caractere=caractere,
                    posicao=idx,
                    bounding_box=bbox,
                    confianca=confianca / 100.0 if confianca > 1 else confianca,
                    imagem_segmento=recorte if self.modo_debug else None
                )
                tabela.adicionar_token(token)
        
        return tabela
    
    def _preparar_recorte_para_ocr(self, recorte: np.ndarray) -> np.ndarray:
        
        # Adiciona borda branca (melhora reconhecimento de caracteres de borda)
        padded = cv2.copyMakeBorder(
            recorte, 10, 10, 10, 10, 
            cv2.BORDER_CONSTANT, 
            value=255
        )
        
        # Redimensiona para altura adequada ao OCR
        h, w = padded.shape[:2]
        if h < 32:
            escala = 32 / h
            novo_w = max(1, int(w * escala))
            padded = cv2.resize(padded, (novo_w, 32), interpolation=cv2.INTER_CUBIC)
        
        return padded
    
    def _extrair_melhor_resultado(
        self, 
        dados_tesseract: dict
    ) -> Tuple[str, float]:
        
        melhor_char = ""
        melhor_conf = 0.0
        
        for i, texto in enumerate(dados_tesseract.get('text', [])):
            texto = texto.strip().upper()
            conf = dados_tesseract.get('conf', [0])[i]
            
            if texto and texto in self.ALFABETO_VALIDO:
                if isinstance(conf, (int, float)) and conf > melhor_conf:
                    melhor_char = texto[0]
                    melhor_conf = float(conf)
        
        return melhor_char, melhor_conf
    
    # =========================================================================
    # FASE 4: ANÁLISE SEMÂNTICA (VALIDAÇÃO)
    # =========================================================================
    
    def _fase_analise_semantica(
        self, 
        tabela_simbolos: TabelaSimbolos
    ) -> TabelaSimbolos:
        
        lexema = tabela_simbolos.obter_lexema_completo()
        
        # Validação de comprimento
        if len(lexema) < 7:
            tabela_simbolos.metadata['erro_comprimento'] = True
            tabela_simbolos.metadata['comprimento_detectado'] = len(lexema)
            return tabela_simbolos
        
        # Trunca para 7 caracteres se necessário
        if len(lexema) > 7:
            lexema = lexema[:7]
            tabela_simbolos.tokens = tabela_simbolos.tokens[:7]
        
        # Padrão esperado de posições: Mercosul (LLLNLNN)
        # True = deve ser letra, False = deve ser dígito
        padrao_posicoes = [True, True, True, False, True, False, False]
        
        tokens_corrigidos = []
        for i, token in enumerate(tabela_simbolos.tokens):
            char_corrigido = token.caractere
            deve_ser_letra = padrao_posicoes[i]
            
            # Aplicação de correção semântica
            if deve_ser_letra and token.caractere.isdigit():
                char_corrigido = self.MAPA_DIGITO_PARA_LETRA.get(
                    token.caractere, token.caractere
                )
                token.confianca *= 0.8  # Penaliza confiança por correção
            elif not deve_ser_letra and token.caractere.isalpha():
                char_corrigido = self.MAPA_LETRA_PARA_DIGITO.get(
                    token.caractere, token.caractere
                )
                token.confianca *= 0.8
            
            # Cria novo token com caractere corrigido
            token_corrigido = Token(
                caractere=char_corrigido,
                posicao=token.posicao,
                bounding_box=token.bounding_box,
                confianca=token.confianca,
                imagem_segmento=token.imagem_segmento
            )
            tokens_corrigidos.append(token_corrigido)
        
        tabela_simbolos.tokens = tokens_corrigidos
        
        # Validação final contra padrões conhecidos
        lexema_final = tabela_simbolos.obter_lexema_completo()
        tabela_simbolos.metadata['padrao_mercosul'] = bool(
            self.PADRAO_MERCOSUL.match(lexema_final)
        )
        tabela_simbolos.metadata['padrao_antigo'] = bool(
            self.PADRAO_ANTIGO.match(lexema_final)
        )
        tabela_simbolos.metadata['valido'] = (
            tabela_simbolos.metadata['padrao_mercosul'] or 
            tabela_simbolos.metadata['padrao_antigo']
        )
        
        return tabela_simbolos
    
    # =========================================================================
    # FASE 5: GERAÇÃO DE CÓDIGO (SÍNTESE)
    # =========================================================================
    
    def _fase_geracao_codigo(
        self, 
        tabela_simbolos: TabelaSimbolos
    ) -> str:
        
        texto_final = tabela_simbolos.obter_lexema_completo()
        
        # Normalização final
        texto_final = texto_final.upper().strip()
        
        # Remove caracteres residuais inválidos
        texto_final = re.sub(r'[^A-Z0-9]', '', texto_final)
        
        return texto_final
    
    # =========================================================================
    # MÉTODO PRINCIPAL: COMPILAÇÃO
    # =========================================================================
    
    def compilar(self, imagem: np.ndarray) -> ResultadoCompilacao:
        
        import time
        inicio = time.perf_counter()
        
        resultado = ResultadoCompilacao(
            sucesso=False,
            texto_final="",
            tabela_simbolos=TabelaSimbolos()
        )
        
        # Validação de entrada
        if imagem is None or imagem.size == 0:
            resultado.erros.append("Imagem de entrada inválida ou vazia")
            return resultado
        
        try:
            # FASE 1: Pré-processamento
            resultado.fases_executadas.append(FaseCompilacao.PRE_PROCESSAMENTO)
            imagem_cinza, imagem_binaria = self._fase_pre_processamento(imagem)
            
            if self.modo_debug:
                resultado.imagens_intermediarias['pre_processamento_cinza'] = imagem_cinza
                resultado.imagens_intermediarias['pre_processamento_binario'] = imagem_binaria
            
            # FASE 2: Análise Léxica (Segmentação)
            resultado.fases_executadas.append(FaseCompilacao.ANALISE_LEXICA)
            caracteres_segmentados = self._fase_analise_lexica(imagem_binaria)
            
            if self.modo_debug:
                resultado.tabela_simbolos.metadata['segmentos_detectados'] = len(caracteres_segmentados)
            
            # Se não segmentou caracteres suficientes, tenta fallback direto
            if len(caracteres_segmentados) < 5:
                resultado.tabela_simbolos = self._fallback_ocr_direto(imagem_binaria)
            else:
                # FASE 3: Análise Sintática (Reconhecimento)
                resultado.fases_executadas.append(FaseCompilacao.ANALISE_SINTATICA)
                resultado.tabela_simbolos = self._fase_analise_sintatica(
                    caracteres_segmentados, imagem_cinza
                )
            
            # FASE 4: Análise Semântica (Validação)
            resultado.fases_executadas.append(FaseCompilacao.ANALISE_SEMANTICA)
            resultado.tabela_simbolos = self._fase_analise_semantica(
                resultado.tabela_simbolos
            )
            
            # FASE 5: Geração de Código (Síntese)
            resultado.fases_executadas.append(FaseCompilacao.GERACAO_CODIGO)
            resultado.texto_final = self._fase_geracao_codigo(
                resultado.tabela_simbolos
            )
            
            # Determina sucesso baseado na validação semântica
            resultado.sucesso = resultado.tabela_simbolos.metadata.get('valido', False)
            
        except Exception as e:
            resultado.erros.append(f"Erro durante compilação: {str(e)}")
        
        # Tempo de execução
        resultado.tempo_execucao_ms = (time.perf_counter() - inicio) * 1000
        
        return resultado
    
    def _fallback_ocr_direto(self, imagem_binaria: np.ndarray) -> TabelaSimbolos:
        
        tabela = TabelaSimbolos()
        tabela.metadata['metodo'] = 'fallback_direto'
        
        try:
            # Adiciona padding
            padded = cv2.copyMakeBorder(
                imagem_binaria, 10, 10, 10, 10,
                cv2.BORDER_CONSTANT, value=255
            )
            
            texto = pytesseract.image_to_string(
                padded,
                lang=self.idiomas,
                config=self._config_linha
            ).strip().upper()
            
            texto = re.sub(r'[^A-Z0-9]', '', texto)
            
            for i, char in enumerate(texto[:7]):
                token = Token(
                    caractere=char,
                    posicao=i,
                    bounding_box=(0, 0, 0, 0),
                    confianca=0.6  # Confiança reduzida por ser fallback
                )
                tabela.adicionar_token(token)
                
        except Exception:
            pass
        
        return tabela
    
    # =========================================================================
    # MÉTODOS AUXILIARES PARA INTERFACE EXTERNA
    # =========================================================================
    
    def ler_placa(self, imagem_processada: np.ndarray) -> str:
        
        resultado = self.compilar(imagem_processada)
        
        if resultado.sucesso and resultado.texto_final:
            return resultado.texto_final[:7]
        
        # Se não validou mas tem texto, retorna mesmo assim (para compatibilidade)
        if resultado.texto_final and len(resultado.texto_final) >= 6:
            return resultado.texto_final[:7]
        
        return ""
    
    def obter_diagnostico(self, resultado: ResultadoCompilacao) -> str:
        
        linhas = [
            "=" * 60,
            "RELATÓRIO DE COMPILAÇÃO OCR",
            "=" * 60,
            f"Sucesso: {'SIM' if resultado.sucesso else 'NÃO'}",
            f"Texto Final: {resultado.texto_final or '(vazio)'}",
            f"Tempo de Execução: {resultado.tempo_execucao_ms:.2f} ms",
            "",
            "FASES EXECUTADAS:",
        ]
        
        for fase in resultado.fases_executadas:
            linhas.append(f"  ✓ {fase.name}")
        
        linhas.extend([
            "",
            "TABELA DE SÍMBOLOS:",
            f"  Tokens: {len(resultado.tabela_simbolos.tokens)}",
            f"  Confiança Média: {resultado.tabela_simbolos.obter_confianca_media():.2%}",
        ])
        
        if resultado.tabela_simbolos.tokens:
            linhas.append("  Detalhes:")
            for token in resultado.tabela_simbolos.tokens:
                linhas.append(
                    f"    [{token.posicao}] '{token.caractere}' "
                    f"({token.tipo}) - {token.confianca:.2%}"
                )
        
        if resultado.erros:
            linhas.extend(["", "ERROS:"])
            for erro in resultado.erros:
                linhas.append(f"  ✗ {erro}")
        
        linhas.append("=" * 60)
        
        return "\n".join(linhas)


# =============================================================================
# EXEMPLO DE USO E TESTES
# =============================================================================

if __name__ == "__main__":
    
    import sys
    
    print("OCRCompilador - Demonstração do Pipeline de Compilação")
    print("=" * 60)
    
    # Criação do compilador em modo debug
    compilador = OCRCompilador(modo_debug=True)
    
    # Exemplo com imagem de teste (se disponível)
    imagem_teste = np.zeros((100, 300), dtype=np.uint8)
    imagem_teste.fill(255)  # Fundo branco
    
    # Simula texto (para demonstração)
    cv2.putText(
        imagem_teste, "ABC1D23", (10, 70),
        cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 3
    )
    
    # Executa compilação
    resultado = compilador.compilar(imagem_teste)
    
    # Exibe diagnóstico
    print(compilador.obter_diagnostico(resultado))
