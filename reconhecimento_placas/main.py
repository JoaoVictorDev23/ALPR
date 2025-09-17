import cv2
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk
from collections import deque
import os
import time

# Módulos do projeto (sem alterações)
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

class PipelineReconhecimento:
    def __init__(self, video_path, detection_interval=5, ocr_interval=15, debug_mode=False):
        self.detector = DetectorPlacaYOLO()  # seu detector original
        self.preprocessador = PreProcessadorROI()
        self.leitor_ocr = LeitorOCR(
            tesseract_cmd=r"C:\\Program Files\\Tesseract-OCR\\tesseract.exe",
            tessdata_dir=r"C:\\Program Files\\Tesseract-OCR\\tessdata",
            langs=("eng", "por")
        )
        self.validador = ValidadorPlaca()

        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            raise IOError(f"Não foi possível abrir o vídeo: {video_path}")
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        self.frame_count = 0
        self.detection_interval = detection_interval
        self.ocr_interval = ocr_interval
        self.debug_mode = debug_mode

        # mantemos a lógica de capturas, mas não desenhamos faixa no vídeo
        self.capturas_recentes = deque(maxlen=6)
        self.placas_ja_capturadas = set()
        self.ocr_cache = {}  # {roi_tuple: (last_frame_idx, placa_validada)}

    def release(self):
        if self.cap:
            self.cap.release()

    def processar_frame(self):
        ok, frame = self.cap.read()
        if not ok:
            return None

        self.frame_count += 1
        resultados = []

        if self.frame_count % self.detection_interval == 0:
            rois = self.detector.detectar(frame)
            for roi in rois:
                x1, y1, x2, y2 = roi
                if x1 >= x2 or y1 >= y2 or min(x2 - x1, y2 - y1) < 8:
                    continue
                placa_roi = frame[y1:y2, x1:x2]
                if placa_roi.size == 0:
                    continue

                last_fi, cached = self.ocr_cache.get(roi, (0, None))
                if self.frame_count - last_fi < self.ocr_interval:
                    resultados.append((roi, cached))
                    continue

                placa_proc = self.preprocessador.processar(placa_roi)
                bruto = self.leitor_ocr.ler_placa(placa_proc)
                validado = self.validador.corrigir_e_validar(bruto)

                if validado and validado not in self.placas_ja_capturadas:
                    conf = self.validador.contador_confianca.get(validado, 0)
                    if conf >= 2:
                        # continua registrando internamente (para a lista à direita),
                        # mas não existe mais a “faixa de capturas” sobre o vídeo.
                        self.capturas_recentes.append({"texto": validado, "thumbnail": placa_roi.copy()})
                        self.placas_ja_capturadas.add(validado)

                self.ocr_cache[roi] = (self.frame_count, validado)
                resultados.append((roi, validado))

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
            filetypes=(("Vídeos", "*.mp4 *.avi *.mov *.mkv *.webm *.m4v"), ("Todos os arquivos", "*.*"))
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

        # === OVERLAY SEM PISCAR: só retângulo quando YOLO detecta ===
        if resultados:
            for roi, placa in resultados:
                x1, y1, x2, y2 = roi
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 255), 2)  # cor fixa, sem efeitos
                if placa:
                    pos_msec = self.pipeline.cap.get(cv2.CAP_PROP_POS_MSEC) or 0
                    self._adicionar_placa_lista(placa, pos_msec)

        # sem faixa de capturas no vídeo
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
