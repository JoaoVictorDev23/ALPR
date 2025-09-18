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

# Módulos do projeto
from detector_placa import DetectorPlacaYOLO
from pre_processador import PreProcessadorROI
from ocr_leitor import LeitorOCR
from validador_placa import ValidadorPlaca

VIDEO_EXTS = (".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v")

def fit_letterbox(frame, target_w, target_h, pad_color=(16, 16, 16)):
    fh, fw = frame.shape[:2]
    scale = min(target_w / fw, target_h / fh)
    new_w, new_h = int(fw * scale), int(fh * scale)
    resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
    top = (target_h - new_h) // 2
    bottom = target_h - new_h - top
    left = (target_w - new_w) // 2
    right = target_w - new_w - left
    return cv2.copyMakeBorder(resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=pad_color)

class TrackedPlate:
    def __init__(self, roi, frame_idx):
        # roi = (x1,y1,x2,y2)
        self.id = str(uuid.uuid4())[:8]
        self.update(roi, frame_idx)
        self.stationary_count = 0
        self.last_moved_frame = frame_idx
        self.ocr_thread_running = False
        self.last_valid = None
        self.last_seen = frame_idx
        self.prev_center = None

    def update(self, roi, frame_idx):
        self.roi = roi
        x1, y1, x2, y2 = roi
        self.center = ((x1 + x2) / 2.0, (y1 + y2) / 2.0)
        self.last_seen = frame_idx

class PipelineReconhecimento:
    def __init__(
        self,
        video_path,
        detection_interval=5,
        ocr_interval=15,
        debug_mode=False,
        stop_frame_threshold=2,      # frames parados para disparar OCR
        stationary_px_thresh=13.0,    # px de tolerância para "parado"
        ocr_max_attempts=4,          # tentativas por thread
        ocr_attempt_delay=0.8,       # segundos entre tentativas
        max_ocr_threads=3            # limite simultâneo de OCR
    ):
        # Componentes
        self.detector = DetectorPlacaYOLO()
        self.preprocessador = PreProcessadorROI()
        self.leitor_ocr = LeitorOCR()
        self.validador = ValidadorPlaca()

        # Vídeo
        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            raise IOError(f"Não foi possível abrir o vídeo: {video_path}")
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        # Parâmetros de operação
        self.frame_count = 0
        self.detection_interval = detection_interval
        self.ocr_interval = ocr_interval
        self.debug_mode = debug_mode

        # Tracking & sincronização
        self.tracked = {}  # id -> TrackedPlate
        self.lock = threading.Lock()

        # Buffer do último frame para uso pelas threads OCR
        self.last_frame = None

        # OCR & captura
        self.capturas_recentes = deque(maxlen=6)
        self.placas_ja_capturadas = set()
        self.ocr_cache = {}
        self.stop_frame_threshold = stop_frame_threshold
        self.stationary_px_thresh = stationary_px_thresh
        self.ocr_max_attempts = ocr_max_attempts
        self.ocr_attempt_delay = ocr_attempt_delay

        # Limpeza de objetos não vistos
        self.max_missing_frames = 30

        # Controle de concorrência para OCR
        self.max_ocr_threads = max_ocr_threads
        self.ocr_sema = threading.Semaphore(self.max_ocr_threads)

    # helper: distância euclidiana
    def _dist(self, a, b):
        return math.hypot(a[0] - b[0], a[1] - b[1])

    # associa detections atuais aos objetos trackeados por centro
    def _associate(self, detections, frame_idx):
        with self.lock:
            assigned = set()
            for det in detections:
                x1, y1, x2, y2 = det
                center = ((x1 + x2) / 2.0, (y1 + y2) / 2.0)
                best_id = None
                best_dist = None
                for tid, tp in self.tracked.items():
                    d = self._dist(center, tp.center)
                    if best_dist is None or d < best_dist:
                        best_dist = d
                        best_id = tid
                # se a melhor distância for pequena o suficiente, atualiza, senão cria novo track
                if best_dist is not None and best_dist < max(self.stationary_px_thresh * 3, 40):
                    # atualiza track existente
                    self.tracked[best_id].update(det, frame_idx)
                    assigned.add(best_id)
                else:
                    # cria novo track
                    new_tp = TrackedPlate(det, frame_idx)
                    self.tracked[new_tp.id] = new_tp
                    assigned.add(new_tp.id)

            # marca objetos não atribuídos (não detectados neste frame)
            for tid in list(self.tracked.keys()):
                if tid not in assigned:
                    # se não visto por muito tempo -> remove (desde que não esteja com OCR ativo)
                    if frame_idx - self.tracked[tid].last_seen > self.max_missing_frames:
                        if not self.tracked[tid].ocr_thread_running:
                            del self.tracked[tid]

    # função OCR rodando em thread por placa (usa self.last_frame)
    def _ocr_worker(self, tid):
        # Limita concorrência
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
                with self.lock:
                    tp = self.tracked.get(tid)
                    frame_ref = self.last_frame.copy() if self.last_frame is not None else None
                    if tp is None or frame_ref is None:
                        break
                    x1, y1, x2, y2 = tp.roi
                    x1, y1, x2, y2 = map(int, (max(0,x1), max(0,y1), max(x2,x1+1), max(y2,y1+1)))
                    placa_roi = frame_ref[y1:y2, x1:x2]

                if placa_roi.size == 0:
                    break

                # Pré-processa e OCR (fora do lock)
                proc = self.preprocessador.processar(placa_roi)
                bruto = self.leitor_ocr.ler_placa(proc)
                valid = self.validador.corrigir_e_validar(bruto)
                attempts += 1

                if valid:
                    conf = self.validador.contador_confianca.get(valid, 0)
                    # aceita se já tem histórico ou após 2 tentativas
                    if conf >= 2 or attempts >= 2:
                        with self.lock:
                            if valid not in self.placas_ja_capturadas:
                                self.capturas_recentes.append({"texto": valid, "thumbnail": placa_roi.copy()})
                                self.placas_ja_capturadas.add(valid)
                            # marca no track
                            tp = self.tracked.get(tid)
                            if tp:
                                tp.last_valid = valid
                        break

                time.sleep(self.ocr_attempt_delay)
        finally:
            with self.lock:
                tp = self.tracked.get(tid)
                if tp:
                    tp.ocr_thread_running = False
            self.ocr_sema.release()

    def release(self):
        if self.cap:
            self.cap.release()

    def processar_frame(self):
        ok, frame = self.cap.read()
        if not ok:
            return None

        # Atualiza buffer do último frame para threads OCR
        with self.lock:
            self.last_frame = frame.copy()

        self.frame_count += 1
        resultados = []

        # Detecta com intervalo (modo leve)
        if self.frame_count % self.detection_interval == 0:
            rois = self.detector.detectar(frame)  # retorna [(x1,y1,x2,y2), ...]
            # filtra rois mínimas
            rois = [r for r in rois if r[2] > r[0] and r[3] > r[1] and min(r[2]-r[0], r[3]-r[1]) >= 12]

            # associa com trackings
            self._associate(rois, self.frame_count)

            # atualiza estado de stationary/movement
            with self.lock:
                for tid, tp in self.tracked.items():
                    if tp.last_seen == self.frame_count:
                        if tp.prev_center is not None:
                            d = self._dist(tp.center, tp.prev_center)
                            if d <= self.stationary_px_thresh:
                                tp.stationary_count += 1
                            else:
                                tp.stationary_count = 0
                                tp.last_moved_frame = self.frame_count
                        tp.prev_center = tp.center

                    # se parado por N frames e não houver thread OCR ativa -> dispara a thread
                    if tp.stationary_count >= self.stop_frame_threshold and not tp.ocr_thread_running:
                        tp.ocr_thread_running = True
                        thr = threading.Thread(target=self._ocr_worker, args=(tid,), daemon=True)
                        thr.start()

                # prepara resultados para UI (rois + última placa validada se houver)
                for tid, tp in self.tracked.items():
                    resultados.append((tp.roi, tp.last_valid))

        # retorno também das capturas recentes (não desenhadas no vídeo)
        return frame, resultados, self.capturas_recentes

class AppReconhecimento:
    WIN_W = 1200
    WIN_H = 720
    VIDEO_W = 960
    VIDEO_H = 540

    def __init__(self, master):
        self.master = master
        self.master.title("Reconhecimento de Placas - TCC")
        self.master.geometry(f"{self.WIN_W}x{self.WIN_H}")
        self.master.resizable(False, False)

        style = ttk.Style(self.master)
        try:
            style.theme_use("clam")
        except:
            pass

        self.video_path = None
        self.pipeline = None
        self.running = False
        self._fps_ui_counter = 0
        self._fps_ui_last = time.time()
        self._ui_fps = 0.0
        self._placas_listadas = set()
        self._canvas_img_id = None

        container = ttk.Frame(self.master, padding=10)
        container.pack(fill="both", expand=True)

        left = ttk.Frame(container)
        left.pack(side="left", fill="y")
        left.config(width=self.VIDEO_W + 10)
        left.pack_propagate(False)

        right = ttk.Frame(container, width=self.WIN_W - self.VIDEO_W - 40)
        right.pack(side="right", fill="both", expand=True)
        right.pack_propagate(True)

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

        # Canvas em pixels (evita sumir botões)
        self.canvas = tk.Canvas(left, width=self.VIDEO_W, height=self.VIDEO_H,
                                bg="#101010", highlightthickness=1, highlightbackground="#2a2a2a")
        self.canvas.pack()

        footer = ttk.Frame(left)
        footer.pack(fill="x", pady=(8, 0))
        self.progress = ttk.Progressbar(
            footer, orient="horizontal", mode="determinate",
            length=self.VIDEO_W - 220, maximum=100
        )
        self.progress.pack(side="left", padx=(0, 8))
        self.lbl_stats = ttk.Label(footer, text="FPS UI: –  |  Frame: –/–  |  Tempo: –")
        self.lbl_stats.pack(side="left")

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
            text="Fluxo: Abrir vídeo → Iniciar → Encerrar. Sem webcam, sem pausa.",
            wraplength=right.winfo_reqwidth()
        ).pack(anchor="w", pady=(8, 0))

        self.master.after(500, self._atualizar_fps_ui)

    # ===== Ações UI =====
    def _selecionar_video(self):
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
        self._loop()

    def _encerrar(self):
        self.running = False
        try:
            if self.pipeline:
                self.pipeline.release()
        except Exception:
            pass
        self.master.destroy()

    def _limpar_lista(self):
        for item in self.tree.get_children():
            self.tree.delete(item)
        self._placas_listadas = set()

    def _adicionar_placa_lista(self, placa, msec):
        if placa in self._placas_listadas:
            return
        self._placas_listadas.add(placa)
        mm = int((msec // 1000) // 60)
        ss = int((msec // 1000) % 60)
        self.tree.insert("", "end", values=(placa, f"{mm:02d}:{ss:02d}"))

    def _loop(self):
        if not self.running:
            return
        saida = self.pipeline.processar_frame()
        if saida is None:
            self._encerrar()
            return

        frame, resultados, _ = saida  # ignoramos a barra de capturas

        # Overlay: pisca em verde quando YOLO detectar algo válido
        if resultados:
            for roi, placa in resultados:
                x1, y1, x2, y2 = roi
                if placa:  # só pisca se já temos OCR válido
                    # controle de piscar: alterna a cada 10 frames
                    if (self.pipeline.frame_count // 10) % 2 == 0:
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
                    pos_msec = self.pipeline.cap.get(cv2.CAP_PROP_POS_MSEC) or 0
                    self._adicionar_placa_lista(placa, pos_msec)

        show = fit_letterbox(frame, self.VIDEO_W, self.VIDEO_H)
        rgb = cv2.cvtColor(show, cv2.COLOR_BGR2RGB)
        imgtk = ImageTk.PhotoImage(Image.fromarray(rgb))
        if self._canvas_img_id is None:
            self._canvas_img_id = self.canvas.create_image(0, 0, image=imgtk, anchor="nw")
        else:
            self.canvas.itemconfig(self._canvas_img_id, image=imgtk)
        self.canvas.image = imgtk

        pos = self.pipeline.cap.get(cv2.CAP_PROP_POS_FRAMES) or 0
        total = self.pipeline.cap.get(cv2.CAP_PROP_FRAME_COUNT) or 1
        msec = self.pipeline.cap.get(cv2.CAP_PROP_POS_MSEC) or 0
        self.progress["value"] = min(100, (pos / total) * 100)

        self._fps_ui_counter += 1
        mm = int((msec // 1000) // 60)
        ss = int((msec // 1000) % 60)
        self.lbl_stats.config(
            text=f"FPS UI: {self._ui_fps:>4.1f}  |  Frame: {int(pos)}/{int(total)}  |  Tempo: {mm:02d}:{ss:02d}"
        )

        self.master.after(30, self._loop)

    def _atualizar_fps_ui(self):
        agora = time.time()
        dt = agora - self._fps_ui_last
        if dt > 0:
            self._ui_fps = self._fps_ui_counter / dt
        self._fps_ui_counter = 0
        self._fps_ui_last = agora
        if self.running:
            self.master.after(500, self._atualizar_fps_ui)

if __name__ == "__main__":
    root = tk.Tk()
    app = AppReconhecimento(root)
    root.mainloop()
