import cv2
import tkinter as tk
from tkinter import filedialog, ttk
from collections import deque
from PIL import Image, ImageTk
import os
import multiprocessing as mp

from detector_placa import DetectorPlacaYOLO
from pre_processador import PreProcessadorROI
from ocr_leitor import LeitorOCR
from validador_placa import ValidadorPlaca
from utils import desenhar_barra_capturas
from utils import desenhar_barra_detectadas

def selecionar_fonte_video():
    root = tk.Tk()
    root.withdraw()
    fonte_selecionada = {"path": None}

    def escolher_video():
        filepath = filedialog.askopenfilename(
            title="Selecione um arquivo de vídeo",
            filetypes=(("Vídeos", "*.mp4 *.avi *.mov"), ("Todos os arquivos", "*.*"))
        )
        if filepath:
            fonte_selecionada["path"] = filepath
        win.destroy()

    def usar_webcam():
        fonte_selecionada["path"] = 0
        win.destroy()

    def fechar():
        fonte_selecionada["path"] = None
        win.destroy()

    win = tk.Toplevel()
    win.title("Fonte de Vídeo")
    win.geometry("300x150")
    win.protocol("WM_DELETE_WINDOW", fechar)

    tk.Label(win, text="Selecione a fonte de vídeo:", font=("Arial", 11)).pack(pady=10)
    tk.Button(win, text="Abrir Arquivo de Vídeo", width=25, command=escolher_video).pack(pady=5)
    tk.Button(win, text="Usar Webcam", width=25, command=usar_webcam).pack(pady=5)

    win.wait_window()
    root.destroy()
    return fonte_selecionada["path"]


def redimensionar_frame(frame, nova_largura=1024):
    altura_original, largura_original = frame.shape[:2]
    proporcao = nova_largura / float(largura_original)
    nova_altura = int(altura_original * proporcao)
    return cv2.resize(frame, (nova_largura, nova_altura), interpolation=cv2.INTER_AREA)


# ---------------- OCR Worker ----------------

def ocr_worker(task_queue, result_queue, tesseract_cmd, tessdata_dir, langs):
    leitor = LeitorOCR(tesseract_cmd, tessdata_dir, langs)
    validador = ValidadorPlaca()
    preprocessador = PreProcessadorROI()

    while True:
        item = task_queue.get()
        if item is None:
            break

        roi_coords, placa_recortada = item
        placa_processada = preprocessador.processar(placa_recortada)
        texto_bruto = leitor.ler_placa(placa_processada)
        placa_validada = validador.corrigir_e_validar(texto_bruto)

        if placa_validada:
            result_queue.put((placa_validada, placa_recortada))


# ---------------- Pipeline Principal ----------------

class PipelineReconhecimento:
    def __init__(self, video_path, detection_interval=5, num_workers=4):
        self.detector = DetectorPlacaYOLO()

        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            raise IOError(f"Não foi possível abrir a fonte de vídeo: {video_path}")

        self.frame_count = 0
        self.detection_interval = detection_interval
        self.last_frame = None
        self.last_detections = []
        self.capturas_recentes = deque(maxlen=5)
        self.placas_ja_capturadas = set()

        # OCR paralelo
        self.task_queue = mp.Queue(maxsize=64)
        self.result_queue = mp.Queue()
        self.workers = []
        for _ in range(num_workers):
            p = mp.Process(
                target=ocr_worker,
                args=(
                    self.task_queue,
                    self.result_queue,
                    r"C:\\Program Files\\Tesseract-OCR\\tesseract.exe",
                    r"C:\\Program Files\\Tesseract-OCR\\tessdata",
                    ("eng", "por"),
                ),
                daemon=True,
            )
            p.start()
            self.workers.append(p)

    def processar_frame(self):
        ret, frame_original = self.cap.read()
        if not ret:
            return None

        self.last_frame = frame_original.copy()
        self.frame_count += 1
        self.last_detections = []

        # YOLO em tempo real
        if self.frame_count % self.detection_interval == 0:
            rois_detectadas = self.detector.detectar(self.last_frame)

            for roi_coords in rois_detectadas:
                x1, y1, x2, y2 = roi_coords
                if x1 >= x2 or y1 >= y2:
                    continue
                placa_recortada = self.last_frame[y1:y2, x1:x2]
                if placa_recortada.size == 0:
                    continue

                # mostra bounding box amarela em tempo real
                self.last_detections.append(roi_coords)

                # manda ROI pro OCR (assíncrono)
                try:
                    self.task_queue.put_nowait((roi_coords, placa_recortada))
                except:
                    pass  # fila cheia, descarta

        # pega resultados do OCR
        while not self.result_queue.empty():
            placa_validada, placa_recortada = self.result_queue.get()
            if placa_validada not in self.placas_ja_capturadas:
                nova_captura = {"texto": placa_validada, "thumbnail": placa_recortada.copy()}
                self.capturas_recentes.append(nova_captura)
                self.placas_ja_capturadas.add(placa_validada)

        return self.last_frame, self.last_detections, self.capturas_recentes

    def liberar(self):
        for _ in self.workers:
            self.task_queue.put(None)
        for p in self.workers:
            p.join()
        self.cap.release()


# ---------------- Tkinter App ----------------

class AppReconhecimento:
    def __init__(self, master, pipeline):
        self.master = master
        self.pipeline = pipeline

        self.master.title("Reconhecimento de Placas - TCC")
        self.master.geometry("1024x700")
        self.master.resizable(False, False)

        self.video_label = tk.Label(self.master)
        self.video_label.pack()

        control_frame = tk.Frame(self.master)
        control_frame.pack(pady=10)

        self.btn_start = ttk.Button(control_frame, text="Iniciar", command=self.iniciar_video)
        self.btn_start.pack(side="left", padx=5)

        self.btn_stop = ttk.Button(control_frame, text="Encerrar", command=self.encerrar)
        self.btn_stop.pack(side="left", padx=5)

        self.running = False

    def iniciar_video(self):
        self.running = True
        self.atualizar_video()

    def encerrar(self):
        self.running = False
        self.pipeline.liberar()
        self.master.destroy()

    def atualizar_video(self):
        if self.running:
            resultado = self.pipeline.processar_frame()
            if resultado is not None:
                frame, detections, capturas = resultado
                frame_para_exibir = redimensionar_frame(frame, nova_largura=1024)
                escala = frame_para_exibir.shape[1] / frame.shape[1]

                # desenha bounding boxes YOLO em amarelo
                for roi in detections:
                    x1, y1, x2, y2 = [int(p * escala) for p in roi]
                    cv2.rectangle(frame_para_exibir, (x1, y1), (x2, y2), (0, 255, 255), 2)

                # desenha barra inferior com OCR já pronto
                # barra superior: YOLO (tempo real)
                desenhar_barra_detectadas(frame_para_exibir, detections)

                # barra inferior: OCR
                desenhar_barra_capturas(frame_para_exibir, list(capturas))

                frame_rgb = cv2.cvtColor(frame_para_exibir, cv2.COLOR_BGR2RGB)
                img = ImageTk.PhotoImage(Image.fromarray(frame_rgb))
                self.video_label.config(image=img)
                self.video_label.image = img

        if self.running:
            self.master.after(30, self.atualizar_video)


if __name__ == "__main__":
    mp.freeze_support()  # necessário no Windows
    caminho_video = selecionar_fonte_video()
    if caminho_video is not None:
        try:
            pipeline = PipelineReconhecimento(video_path=caminho_video, detection_interval=5, num_workers=4)
            root = tk.Tk()
            app = AppReconhecimento(root, pipeline)
            root.mainloop()
        except Exception as e:
            print(f"Ocorreu um erro: {e}")
    else:
        print("Nenhuma fonte de vídeo selecionada. Encerrando o programa.")
