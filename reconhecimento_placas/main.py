# main.py
import cv2
import tkinter as tk
from tkinter import filedialog
from collections import deque
from detector_placa import DetectorPlacaYOLO
from pre_processador import PreProcessadorROI
from ocr_leitor import LeitorOCR
from validador_placa import ValidadorPlaca
from utils import desenhar_resultados, desenhar_barra_capturas
# Adicione no início do main.py
import os


def salvar_debug(placa_recortada, placa_processada, texto_bruto, texto_validado, frame_count):
    """Salva imagens para debug"""
    debug_dir = "debug"
    if not os.path.exists(debug_dir):
        os.makedirs(debug_dir)

    cv2.imwrite(f"{debug_dir}/frame_{frame_count}_original.jpg", placa_recortada)
    cv2.imwrite(f"{debug_dir}/frame_{frame_count}_processada.jpg", placa_processada)

    with open(f"{debug_dir}/frame_{frame_count}_texto.txt", "w") as f:
        f.write(f"Bruto: {texto_bruto}\n")
        f.write(f"Validado: {texto_validado}\n")
# ... (a função selecionar_fonte_video e redimensionar_frame continuam as mesmas) ...
def selecionar_fonte_video():
    root = tk.Tk()
    root.withdraw()
    selection_window = tk.Toplevel(root)
    selection_window.title("Fonte de Vídeo")
    selection_window.geometry("350x180")
    selection_window.resizable(False, False)
    fonte_selecionada = {"path": None}
    def escolher_video():
        filepath = filedialog.askopenfilename(title="Selecione um arquivo de vídeo", filetypes=(("Vídeos", "*.mp4 *.avi *.mov"), ("Todos os arquivos", "*.*")))
        if filepath:
            fonte_selecionada["path"] = filepath
            selection_window.destroy()
    def usar_webcam():
        fonte_selecionada["path"] = 0
        selection_window.destroy()
    def on_closing():
        fonte_selecionada["path"] = None
        selection_window.destroy()
    selection_window.protocol("WM_DELETE_WINDOW", on_closing)
    label = tk.Label(selection_window, text="Selecione a fonte para o reconhecimento:", pady=10, font=("Arial", 10))
    label.pack()
    btn_video = tk.Button(selection_window, text="Abrir Arquivo de Vídeo", command=escolher_video, width=30, height=2, font=("Arial", 10))
    btn_video.pack(pady=10)
    btn_webcam = tk.Button(selection_window, text="Usar Webcam em Tempo Real", command=usar_webcam, width=30, height=2, font=("Arial", 10))
    btn_webcam.pack(pady=5)
    selection_window.update_idletasks()
    width, height = selection_window.winfo_width(), selection_window.winfo_height()
    x = (selection_window.winfo_screenwidth() // 2) - (width // 2)
    y = (selection_window.winfo_screenheight() // 2) - (height // 2)
    selection_window.geometry(f'{width}x{height}+{x}+{y}')
    selection_window.wait_window()
    root.destroy()
    return fonte_selecionada["path"]

def redimensionar_frame(frame, nova_largura=1024):
    altura_original, largura_original = frame.shape[:2]
    if largura_original <= nova_largura:
        return frame
    proporcao = nova_largura / float(largura_original)
    nova_altura = int(altura_original * proporcao)
    dimensoes = (nova_largura, nova_altura)
    return cv2.resize(frame, dimensoes, interpolation=cv2.INTER_AREA)


class PipelineReconhecimento:
    def __init__(self, video_path):
        # Componentes do pipeline
        self.detector = DetectorPlacaYOLO()
        self.preprocessador = PreProcessadorROI()
        self.leitor_ocr = LeitorOCR()  # EasyOCR será inicializado aqui
        self.validador = ValidadorPlaca()
        
        # Fonte de vídeo
        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            raise IOError(f"Não foi possível abrir a fonte de vídeo: {video_path}")
        
        # --- NOVOS ATRIBUTOS DE ESTADO ---
        self.is_paused = False
        self.frame_count = 0
        self.detection_interval = 5  # Roda a detecção a cada 5 frames
        self.last_results = []       # Armazena os últimos resultados válidos

        # Fila para o histórico de capturas (limite de 5 placas)
        self.capturas_recentes = deque(maxlen=5)
        self.placas_ja_capturadas = set() # Para evitar capturas duplicadas

        print("Sistema iniciado. Pressione 'ESPACO' para pausar/continuar e 'q' para sair.")

    def executar(self):
        while True:
            # Lógica de Pausa: só lê um novo frame se não estiver pausado
            if not self.is_paused:
                ret, frame_original = self.cap.read()
                if not ret:
                    print("Fim do vídeo ou erro na captura.")
                    break
                
                # Armazena o último frame lido para ser exibido durante a pausa
                self.last_frame = frame_original.copy()
                self.frame_count += 1
            
            frame_para_exibir = redimensionar_frame(self.last_frame, nova_largura=1024)
            escala = frame_para_exibir.shape[1] / self.last_frame.shape[1]

            # --- LÓGICA DE OTIMIZAÇÃO DE FPS ---
            # Roda a detecção completa apenas em intervalos
            if not self.is_paused:  # Detecta em TODOS os frames para testar
                self.last_results = [] # Limpa resultados antigos antes de uma nova detecção
                rois_detectadas = self.detector.detectar(self.last_frame)

                # No loop principal do main.py, adicione prints de debug:
                for roi_coords in rois_detectadas:
                    x1, y1, x2, y2 = roi_coords
                    print(f"ROI detectada: {x1},{y1} - {x2},{y2}")

                    if x1 >= x2 or y1 >= y2:
                        print("ROI inválida - pulando")
                        continue

                    placa_recortada = self.last_frame[y1:y2, x1:x2]

                    if placa_recortada.size == 0:
                        print("ROI vazia - pulando")
                        continue

                    # DEBUG: Salva a ROI original
                    cv2.imwrite(f"debug_roi_{self.frame_count}.jpg", placa_recortada)

                    placa_processada = self.preprocessador.processar(placa_recortada)

                    # DEBUG: Salva a ROI processada
                    cv2.imwrite(f"debug_processed_{self.frame_count}.jpg", placa_processada)

                    texto_bruto = self.leitor_ocr.ler_placa(placa_processada)
                    print(f"Texto bruto do OCR: '{texto_bruto}'")

                    placa_validada = self.validador.corrigir_e_validar(texto_bruto)
                    print(f"Texto validado: '{placa_validada}'")

                    self.last_results.append((roi_coords, placa_validada))
                    # --- LÓGICA DE CAPTURA DE PLACA ---
                    if placa_validada and placa_validada not in self.placas_ja_capturadas:
                        nova_captura = {
                            "texto": placa_validada,
                            "thumbnail": self.last_frame[y1:y2, x1:x2].copy()
                        }
                        self.capturas_recentes.append(nova_captura)
                        self.placas_ja_capturadas.add(placa_validada)
                        print(f"Nova placa capturada: {placa_validada}")

            # Desenha os últimos resultados conhecidos em todos os frames
            for roi, placa in self.last_results:
                roi_ajustada = [int(p * escala) for p in roi]
                desenhar_resultados(frame_para_exibir, roi_ajustada, placa)

            # Desenha a barra de capturas
            desenhar_barra_capturas(frame_para_exibir, self.capturas_recentes)

            # Mostra texto de "PAUSADO" se for o caso
            if self.is_paused:
                cv2.putText(frame_para_exibir, "PAUSADO", (30, 50), 
                            cv2.FONT_HERSHEY_TRIPLEX, 1.5, (0, 200, 255), 2)

            cv2.imshow('Reconhecimento de Placas Veiculares - TCC', frame_para_exibir)

            # Controle de Teclas
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            if key == ord(' '): # Barra de espaço
                self.is_paused = not self.is_paused

        self.cap.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    caminho_video = selecionar_fonte_video()
    if caminho_video is not None:
        try:
            pipeline = PipelineReconhecimento(video_path=caminho_video)
            pipeline.executar()
        except Exception as e:
            print(f"Ocorreu um erro fatal na execução: {e}")
    else:
        print("Nenhuma fonte de vídeo selecionada. Encerrando o programa.")