import cv2
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk
from collections import deque
import os
import time
import threading
import math
import uuid

# =============================
# Módulos do projeto (componentes da pipeline):
# - DetectorPlacaYOLO: detecta placas no frame (retorna caixas/ROIs).
# - PreProcessadorROI: melhora a ROI para o OCR (ex.: equalização, binarização).
# - OCRCompilador: reconhecimento de caracteres estruturado como compilador.
# - ValidadorPlaca: corrige e valida o texto reconhecido (regex/heurísticas).
#
# NOTA ACADÊMICA: O módulo OCRCompilador implementa o reconhecimento óptico
# de caracteres utilizando arquitetura inspirada em compiladores, permitindo
# explicação teórica das fases de Análise Léxica, Sintática e Semântica.
# =============================
from detector_placa import DetectorPlacaYOLO
from pre_processador import PreProcessadorROI
from ocr_compilador import OCRCompilador  # Substituição: LeitorOCR -> OCRCompilador
from validador_placa import ValidadorPlaca

# Extensões de vídeo válidas para seleção via diálogo
VIDEO_EXTS = (".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v")

def fit_letterbox(frame, target_w, target_h, pad_color=(16, 16, 16)):
    """
    Redimensiona o frame mantendo proporção (sem distorcer) e adiciona bordas
    (letterbox) para preencher exatamente o tamanho alvo (target_w x target_h).
    """
    fh, fw = frame.shape[:2]
    scale = min(target_w / fw, target_h / fh)  # fator de escala que preserva aspecto
    new_w, new_h = int(fw * scale), int(fh * scale)
    resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
    # calcula bordas (superior/inferior/esquerda/direita) para centralizar a imagem
    top = (target_h - new_h) // 2
    bottom = target_h - new_h - top
    left = (target_w - new_w) // 2
    right = target_w - new_w - left
    # adiciona bordas com cor padronizada (cinza escuro)
    return cv2.copyMakeBorder(resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=pad_color)

class TrackedPlate:
    """
    Representa uma 'trilha' (track) para uma placa detectada.
    Mantém estado entre frames: ROI, centro, contagem parado, última validação etc.
    """
    def __init__(self, roi, frame_idx):
        # roi = (x1,y1,x2,y2)
        self.id = str(uuid.uuid4())[:8]   # identificador curto para a trilha
        self.update(roi, frame_idx)       # inicializa centro/last_seen com a ROI
        self.stationary_count = 0         # quantos frames consecutivos a ROI ficou "parada"
        self.last_moved_frame = frame_idx # último frame em que se moveu
        self.ocr_thread_running = False   # evita 2 threads de OCR simultâneas nesse track
        self.last_valid = None            # último texto de placa validado pelo OCR
        self.last_seen = frame_idx        # último frame em que foi vista
        self.prev_center = None           # centro no frame anterior (para medir deslocamento)

    def update(self, roi, frame_idx):
        """
        Atualiza ROI e centro; marca que a trilha foi vista neste frame.
        """
        self.roi = roi
        x1, y1, x2, y2 = roi
        self.center = ((x1 + x2) / 2.0, (y1 + y2) / 2.0)  # centro geométrico da caixa
        self.last_seen = frame_idx

class PipelineReconhecimento:
    """
    Orquestra o fluxo completo:
    - lê frames do vídeo
    - detecta placas em intervalo de frames
    - associa detecções a trilhas existentes por proximidade de centro
    - dispara threads de OCR quando a trilha fica "parada"
    - valida/guarda resultados e devolve dados para a UI desenhar
    """
    def __init__(
        self,
        video_path,
        detection_interval=5,
        ocr_interval=15,
        debug_mode=False,
        stop_frame_threshold=2,      # frames consecutivos "parado" para disparar OCR
        stationary_px_thresh=13.0,   # tolerância em pixels para considerar "parado"
        ocr_max_attempts=4,          # tentativas de OCR por thread/trilha
        ocr_attempt_delay=0.8,       # intervalo entre tentativas de OCR (segundos)
        max_ocr_threads=3            # número máximo de threads de OCR simultâneas
    ):
        # Componentes principais da pipeline (detector, pré-processador, OCR-Compilador, validador)
        self.detector = DetectorPlacaYOLO()
        self.preprocessador = PreProcessadorROI()
        self.leitor_ocr = OCRCompilador()  # OCRCompilador substitui LeitorOCR
        self.validador = ValidadorPlaca()

        # Fonte de vídeo
        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            raise IOError(f"Não foi possível abrir o vídeo: {video_path}")
        # Buffer pequeno reduz latência de leitura (quando aplicável)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        # Parâmetros de operação
        self.frame_count = 0                  # contador de frames processados
        self.detection_interval = detection_interval  # detecta a cada N frames (economia de CPU/GPU)
        self.ocr_interval = ocr_interval      # (reservado/não usado diretamente aqui)
        self.debug_mode = debug_mode

        # Estruturas de tracking e sincronização
        self.tracked = {}              # dicionário: id -> TrackedPlate
        self.lock = threading.Lock()   # protege acesso a 'tracked' e 'last_frame'

        # Buffer do último frame (snapshot para threads de OCR)
        self.last_frame = None

        # OCR & captura (saídas e controle)
        self.capturas_recentes = deque(maxlen=6)  # últimas placas válidas (texto + thumbnail)
        self.placas_ja_capturadas = set()         # evita duplicar placas já aceitas
        self.ocr_cache = {}                        # reservado para caches de OCR (não utilizado aqui)
        self.stop_frame_threshold = stop_frame_threshold
        self.stationary_px_thresh = stationary_px_thresh
        self.ocr_max_attempts = ocr_max_attempts
        self.ocr_attempt_delay = ocr_attempt_delay

        # Remoção de trilhas não vistas por muito tempo (se não estiverem em OCR)
        self.max_missing_frames = 30

        # Controle de concorrência para OCR (limita threads de OCR ativas)
        self.max_ocr_threads = max_ocr_threads
        self.ocr_sema = threading.Semaphore(self.max_ocr_threads)

    # helper: distância euclidiana
    def _dist(self, a, b):
        """
        Calcula distância euclidiana entre dois pontos 2D a=(x,y) e b=(x,y).
        Usado para associar detecções às trilhas e para medir movimento.
        """
        return math.hypot(a[0] - b[0], a[1] - b[1])

    # associa detections atuais aos objetos trackeados por centro
    def _associate(self, detections, frame_idx):
        """
        Associa detecções (ROIs do YOLO) às trilhas existentes:
        - para cada detecção, encontra a trilha mais próxima em termos de centro;
        - se a distância for pequena o suficiente, atualiza essa trilha;
        - caso contrário, cria uma nova trilha.
        - também remove trilhas que ficaram 'sumidas' por muito tempo (sem OCR rodando).
        """
        with self.lock:
            assigned = set()
            for det in detections:
                x1, y1, x2, y2 = det
                center = ((x1 + x2) / 2.0, (y1 + y2) / 2.0)  # centro da detecção atual
                best_id = None
                best_dist = None
                # procura a trilha com centro mais próximo
                for tid, tp in self.tracked.items():
                    d = self._dist(center, tp.center)
                    if best_dist is None or d < best_dist:
                        best_dist = d
                        best_id = tid
                # se perto o bastante, assume que é a mesma placa e atualiza
                if best_dist is not None and best_dist < max(self.stationary_px_thresh * 3, 40):
                    # atualiza track existente
                    self.tracked[best_id].update(det, frame_idx)
                    assigned.add(best_id)
                else:
                    # cria novo track para essa detecção
                    new_tp = TrackedPlate(det, frame_idx)
                    self.tracked[new_tp.id] = new_tp
                    assigned.add(new_tp.id)

            # marca trilhas não atribuídas neste frame (não houve detecção correspondente)
            for tid in list(self.tracked.keys()):
                if tid not in assigned:
                    # se não vista por muito tempo -> remove (desde que não esteja com OCR ativo)
                    if frame_idx - self.tracked[tid].last_seen > self.max_missing_frames:
                        if not self.tracked[tid].ocr_thread_running:
                            del self.tracked[tid]

    # função OCR rodando em thread por placa (usa self.last_frame)
    def _ocr_worker(self, tid):
        """
        Thread de OCR para uma trilha específica (id = tid).
        Fluxo:
        - tenta adquirir semáforo (limita quantas threads de OCR rodam ao mesmo tempo);
        - captura snapshot do último frame e recorta a ROI da trilha;
        - pré-processa a ROI, aplica OCR e valida/corrige o texto;
        - se obtiver leitura válida e confiável, salva captura e marca em last_valid;
        - repete algumas tentativas com atrasos curtos (para tentar um frame melhor).
        """
        # Limita concorrência (se não conseguir, desarma flag e sai)
        acquired = self.ocr_sema.acquire(timeout=0.1)
        if not acquired:
            with self.lock:
                tp = self.tracked.get(tid)
                if tp:
                    tp.ocr_thread_running = False
            return

        try:
            attempts = 0
            while attempts < self.ocr_max_attempts:
                # BLOCO CRÍTICO: lê estado compartilhado com segurança
                with self.lock:
                    tp = self.tracked.get(tid)
                    # snapshot do último frame para evitar race condition de leitura
                    frame_ref = self.last_frame.copy() if self.last_frame is not None else None
                    if tp is None or frame_ref is None:
                        break
                    x1, y1, x2, y2 = tp.roi
                    # garante índices válidos e área mínima (pelo menos 1 px)
                    x1, y1, x2, y2 = map(int, (max(0,x1), max(0,y1), max(x2,x1+1), max(y2,y1+1)))
                    placa_roi = frame_ref[y1:y2, x1:x2]

                # Se ROI vazia (por recorte mínimo/bordas), aborta
                if placa_roi.size == 0:
                    break

                # Pré-processa e OCR (fora do lock — parte pesada)
                proc = self.preprocessador.processar(placa_roi)
                bruto = self.leitor_ocr.ler_placa(proc)
                valid = self.validador.corrigir_e_validar(bruto)
                attempts += 1

                if valid:
                    # contador/heurística de confiança do validador (consistência)
                    conf = self.validador.contador_confianca.get(valid, 0)
                    # aceita se já tem histórico suficiente OU após >= 2 tentativas
                    if conf >= 2 or attempts >= 2:
                        with self.lock:
                            if valid not in self.placas_ja_capturadas:
                                # guarda texto e miniatura da ROI original
                                self.capturas_recentes.append({"texto": valid, "thumbnail": placa_roi.copy()})
                                self.placas_ja_capturadas.add(valid)
                            # marca no track a última leitura válida
                            tp = self.tracked.get(tid)
                            if tp:
                                tp.last_valid = valid
                        break

                # Pequeno atraso entre tentativas (aguarda possível frame mais nítido)
                time.sleep(self.ocr_attempt_delay)
        finally:
            # Em qualquer saída, limpa estado e libera vaga no semáforo
            with self.lock:
                tp = self.tracked.get(tid)
                if tp:
                    tp.ocr_thread_running = False
            self.ocr_sema.release()

    def release(self):
        """
        Libera a fonte de vídeo (boa prática ao encerrar).
        """
        if self.cap:
            self.cap.release()

    def processar_frame(self):
        """
        Processa um único frame do vídeo:
        - lê o próximo frame (ou retorna None no fim);
        - atualiza snapshot do último frame (para OCR);
        - a cada N frames, roda detecção e associa trilhas;
        - atualiza estado "parado" das trilhas e dispara threads de OCR quando necessário;
        - retorna (frame, resultados, capturas_recentes) para a UI.
        """
        ok, frame = self.cap.read()
        if not ok:
            return None  # fim do vídeo ou erro

        # Atualiza buffer do último frame para threads OCR (snapshot protegido)
        with self.lock:
            self.last_frame = frame.copy()

        self.frame_count += 1
        resultados = []  # lista de tuplas (roi, last_valid) para a UI

        # Detecta com intervalo (alivia custo de detecção)
        if self.frame_count % self.detection_interval == 0:
            rois = self.detector.detectar(frame)  # retorna [(x1,y1,x2,y2), ...]
            # filtra rois mínimas (evita caixas degeneradas)
            rois = [r for r in rois if r[2] > r[0] and r[3] > r[1] and min(r[2]-r[0], r[3]-r[1]) >= 12]

            # associa com trackings existentes (ou cria novos)
            self._associate(rois, self.frame_count)

            # atualiza estado de stationary/movement e dispara OCR quando parado
            with self.lock:
                for tid, tp in self.tracked.items():
                    if tp.last_seen == self.frame_count:
                        # calcula deslocamento do centro entre frames
                        if tp.prev_center is not None:
                            d = self._dist(tp.center, tp.prev_center)
                            if d <= self.stationary_px_thresh:
                                tp.stationary_count += 1  # permaneceu parado dentro da tolerância
                            else:
                                tp.stationary_count = 0   # houve movimento
                                tp.last_moved_frame = self.frame_count
                        tp.prev_center = tp.center

                    # Se parado por N frames e sem OCR ativo -> dispara thread de OCR
                    if tp.stationary_count >= self.stop_frame_threshold and not tp.ocr_thread_running:
                        tp.ocr_thread_running = True
                        thr = threading.Thread(target=self._ocr_worker, args=(tid,), daemon=True)
                        thr.start()

                # prepara resultados para UI (desenho + última placa validada, se existir)
                for tid, tp in self.tracked.items():
                    resultados.append((tp.roi, tp.last_valid))

        # retorno também das capturas recentes (não desenhadas no vídeo)
        return frame, resultados, self.capturas_recentes

class AppReconhecimento:
    # Dimensões base da janela e do canvas de vídeo (já aumentadas)
    WIN_W = 1400
    WIN_H = 820
    VIDEO_W = 1120
    VIDEO_H = 630

    def __init__(self, master):
        """
        Constrói a interface Tkinter:
        - lado esquerdo: seleção de vídeo, canvas para exibição e rodapé com status;
        - lado direito: lista (Treeview) com placas reconhecidas (texto + tempo).
        """
        self.master = master
        self.master.title("Reconhecimento de Placas - TCC")
        self.master.geometry(f"{self.WIN_W}x{self.WIN_H}")
        self.master.resizable(False, False)

        # Estilo ttk (tema visual)
        style = ttk.Style(self.master)
        try:
            style.theme_use("clam")
        except:
            pass

        # Estado da aplicação
        self.video_path = None
        self.pipeline = None
        self.running = False
        self._fps_ui_counter = 0
        self._fps_ui_last = time.time()
        self._ui_fps = 0.0
        self._placas_listadas = set()  # evita repetir a mesma placa na lista
        self._canvas_img_id = None     # id do item de imagem no Canvas (para atualizar sem recriar)

        # Container principal (margem/padding)
        container = ttk.Frame(self.master, padding=10)
        container.pack(fill="both", expand=True)

        # Lado esquerdo: vídeo e controles
        left = ttk.Frame(container)
        left.pack(side="left", fill="y")
        left.config(width=self.VIDEO_W + 10)
        left.pack_propagate(False)

        # Lado direito: lista de resultados
        right = ttk.Frame(container, width=self.WIN_W - self.VIDEO_W - 40)
        right.pack(side="right", fill="both", expand=True)
        right.pack_propagate(True)

        # Cabeçalho (esquerda): label do arquivo e botões
        header = ttk.Frame(left)
        header.pack(fill="x", pady=(0, 8))

        self.lbl_video = ttk.Label(header, text="Nenhum vídeo selecionado", width=60)
        self.lbl_video.pack(side="left")

        btns = ttk.Frame(header)
        btns.pack(side="right")
        self.btn_select = ttk.Button(btns, text="Abrir vídeo…", command=self._selecionar_video)
        self.btn_select.pack(side="left", padx=(0, 6))
        self.btn_start = ttk.Button(btns, text="Iniciar", command=self._iniciar, state="disabled")
        self.btn_start.pack(side="left", padx=(0, 6))
        self.btn_stop = ttk.Button(btns, text="Encerrar", command=self._encerrar)
        self.btn_stop.pack(side="left")

        # Canvas em pixels (mantém tamanho fixo, evita reposicionar botões)
        self.canvas = tk.Canvas(left, width=self.VIDEO_W, height=self.VIDEO_H,
                                bg="#101010", highlightthickness=1, highlightbackground="#2a2a2a")
        self.canvas.pack()

        # Rodapé (esquerda): progresso + status (frame/tempo)
        footer = ttk.Frame(left)
        footer.pack(fill="x", pady=(8, 0))
        self.progress = ttk.Progressbar(
            footer, orient="horizontal", mode="determinate",
            length=self.VIDEO_W - 220, maximum=100
        )
        self.progress.pack(side="left", padx=(0, 8))
        self.lbl_stats = ttk.Label(footer, text="Frame: –/–  |  Tempo: –")
        self.lbl_stats.pack(side="left")

        # Lado direito: tabela de detecções (placa + timestamp)
        ttk.Label(right, text="Detecções (placas reconhecidas)", font=("Segoe UI", 10, "bold")).pack(anchor="w")
        columns = ("placa", "momento")
        self.tree = ttk.Treeview(right, columns=columns, show="headings", height=20)
        self.tree.heading("placa", text="Placa")
        self.tree.heading("momento", text="Momento (mm:ss)")
        self.tree.column("placa", width=120, anchor="center")
        self.tree.column("momento", width=120, anchor="center")
        self.tree.pack(fill="both", expand=True, pady=(6, 0))

        ttk.Label(
            right,
            text="Fluxo: Abrir vídeo → Iniciar → Encerrar.",
            wraplength=right.winfo_reqwidth()
        ).pack(anchor="w", pady=(8, 0))

        # Timer para atualizar métrica de FPS da UI (mesmo que oculto do texto)
        self.master.after(500, self._atualizar_fps_ui)

    # ===== Ações UI =====
    def _selecionar_video(self):
        """
        Abre diálogo para selecionar arquivo de vídeo, valida extensão e
        habilita o botão 'Iniciar' quando válido.
        """
        path = filedialog.askopenfilename(
            title="Selecione um arquivo de vídeo",
            filetypes=(("Vídeos", "*.mp4 *.avi *.mov *.mkv *.webm, *.m4v"), ("Todos os arquivos", "*.*"))
        )
        if not path:
            return
        if not path.lower().endswith(VIDEO_EXTS):
            messagebox.showerror("Arquivo inválido",
                                 "Selecione um arquivo de vídeo (.mp4, .avi, .mov, .mkv, .webm, .m4v).")
            return
        self.video_path = path
        self.lbl_video.config(text=os.path.basename(path))
        self.btn_start.config(state="normal")

    def _iniciar(self):
        """
        Inicializa (ou reinicializa) a pipeline com o vídeo escolhido e
        inicia o loop de processamento/atualização da UI.
        """
        if self.running:
            return
        if not self.video_path:
            messagebox.showwarning("Seleção necessária", "Escolha um vídeo antes de iniciar.")
            return
        try:
            if self.pipeline:
                self.pipeline.release()
        except Exception:
            pass
        self.pipeline = PipelineReconhecimento(
            video_path=self.video_path,
            detection_interval=5,
            ocr_interval=15,
            debug_mode=False
        )
        self._limpar_lista()
        self.running = True
        self._loop()  # agenda o loop de UI/processing

    def _encerrar(self):
        """
        Encerra o loop, libera o vídeo e, se houver leituras, pergunta se deve
        salvar (TXT) ou enviar ao endpoint. Ao final, fecha a janela.
        """
        self.running = False
        try:
            if self.pipeline:
                self.pipeline.release()
        except Exception:
            pass

        # Popup perguntando se deseja salvar
        if self.pipeline and self.pipeline.capturas_recentes:
            resposta = messagebox.askyesno(
                "Salvar placas",
                "Deseja salvar as placas reconhecidas?"
            )
            if resposta:
                self._mostrar_opcoes_salvamento()

        self.master.destroy()

    def _mostrar_opcoes_salvamento(self):
        """Cria popup com botões de escolha do método de salvamento."""
        win = tk.Toplevel(self.master)
        win.title("Escolha o método de salvamento")
        win.geometry("300x120")
        win.resizable(False, False)

        ttk.Label(win, text="Selecione o método:", font=("Segoe UI", 10, "bold")).pack(pady=10)

        btn_frame = ttk.Frame(win)
        btn_frame.pack(pady=5)

        ttk.Button(
            btn_frame,
            text="Salvar TXT",
            command=lambda: [self._salvar_txt(), win.destroy()]
        ).pack(side="left", padx=15)

        ttk.Button(
            btn_frame,
            text="Enviar Endpoint",
            command=lambda: [self._enviar_endpoint(), win.destroy()]
        ).pack(side="right", padx=15)

    def _salvar_txt(self):
        """Salva todas as placas detectadas em um arquivo .txt (uma por linha)."""
        import os
        pasta = "placas_salvas"
        os.makedirs(pasta, exist_ok=True)
        caminho = os.path.join(pasta, "placas_detectadas.txt")

        with open(caminho, "w", encoding="utf-8") as f:
            for captura in self.pipeline.capturas_recentes:
                f.write(captura["texto"] + "\n")

        messagebox.showinfo("Sucesso", f"Placas salvas em {caminho}")

    def _enviar_endpoint(self):
        """Envia imagens das placas para o endpoint configurado (ex.: API local)."""
        import requests, cv2
        url = "http://localhost:8000/salvar_placa/"
        for captura in self.pipeline.capturas_recentes:
            # codifica miniatura da ROI como JPEG em memória
            _, buffer = cv2.imencode(".jpg", captura["thumbnail"])
            requests.post(
                url,
                data={"placa": captura["texto"]},
                files={"file": (f"{captura['texto']}.jpg", buffer.tobytes(), "image/jpeg")}
            )
        messagebox.showinfo("Sucesso", "Placas enviadas para o endpoint.")

    def _limpar_lista(self):
        """Apaga todos os itens da tabela e zera o conjunto de já listadas."""
        for item in self.tree.get_children():
            self.tree.delete(item)
        self._placas_listadas = set()

    def _adicionar_placa_lista(self, placa, msec):
        """
        Insere uma linha na tabela com (placa, mm:ss), evitando duplicatas
        na sessão (uma placa só aparece uma vez).
        """
        if placa in self._placas_listadas:
            return
        self._placas_listadas.add(placa)
        mm = int((msec // 1000) // 60)
        ss = int((msec // 1000) % 60)
        self.tree.insert("", "end", values=(placa, f"{mm:02d}:{ss:02d}"))

    def _loop(self):
        """
        Loop principal da UI:
        - pede um frame processado à pipeline;
        - desenha overlay (caixa verde piscante quando há OCR válido);
        - atualiza imagem no Canvas, barra de progresso e status (frame/tempo);
        - reagenda a próxima iteração após ~30ms (~33 FPS de UI).
        """
        if not self.running:
            return
        saida = self.pipeline.processar_frame()
        if saida is None:
            self._encerrar()
            return

        frame, resultados, _ = saida  # ignoramos a barra de capturas aqui

        # Overlay: desenha retângulo verde piscante para ROIs com OCR válido
        if resultados:
            for roi, placa in resultados:
                x1, y1, x2, y2 = roi
                if placa:  # só pisca se já temos OCR válido
                    # alterna a cada 10 frames para efeito de piscar
                    if (self.pipeline.frame_count // 10) % 2 == 0:
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
                    pos_msec = self.pipeline.cap.get(cv2.CAP_PROP_POS_MSEC) or 0
                    self._adicionar_placa_lista(placa, pos_msec)

        # Ajusta o frame ao Canvas com letterbox e converte BGR->RGB para exibir no Tk
        show = fit_letterbox(frame, self.VIDEO_W, self.VIDEO_H)
        rgb = cv2.cvtColor(show, cv2.COLOR_BGR2RGB)
        imgtk = ImageTk.PhotoImage(Image.fromarray(rgb))
        # Atualiza imagem no Canvas (mantém referência para evitar coleta do objeto)
        if self._canvas_img_id is None:
            self._canvas_img_id = self.canvas.create_image(0, 0, image=imgtk, anchor="nw")
        else:
            self.canvas.itemconfig(self._canvas_img_id, image=imgtk)
        self.canvas.image = imgtk

        # Atualiza barra de progresso e status (frame atual, total e tempo mm:ss)
        pos = self.pipeline.cap.get(cv2.CAP_PROP_POS_FRAMES) or 0
        total = self.pipeline.cap.get(cv2.CAP_PROP_FRAME_COUNT) or 1
        msec = self.pipeline.cap.get(cv2.CAP_PROP_POS_MSEC) or 0
        self.progress["value"] = min(100, (pos / total) * 100)

        mm = int((msec // 1000) // 60)
        ss = int((msec // 1000) % 60)
        self.lbl_stats.config(
            text=f"Frame: {int(pos)}/{int(total)}  |  Tempo: {mm:02d}:{ss:02d}"
        )

        # Reagenda próxima iteração da UI
        self.master.after(30, self._loop)

    def _atualizar_fps_ui(self):
        """
        Calcula FPS médio da UI a cada 500ms (mantido para telemetria interna,
        embora não esteja sendo exibido no rótulo).
        """
        agora = time.time()
        dt = agora - self._fps_ui_last
        if dt > 0:
            self._ui_fps = self._fps_ui_counter / dt
        self._fps_ui_counter = 0
        self._fps_ui_last = agora
        if self.running:
            self.master.after(500, self._atualizar_fps_ui)

if __name__ == "__main__":
    # Cria a janela raiz e inicia o app
    root = tk.Tk()
    app = AppReconhecimento(root)
    root.mainloop()
