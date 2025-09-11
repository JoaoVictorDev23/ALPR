# debug.py
import cv2
from detector_placa import DetectorPlacaYOLO
from pre_processador import PreProcessadorROI
from ocr_leitor import LeitorOCR
from validador_placa import ValidadorPlaca

# Teste com um frame específico
cap = cv2.VideoCapture("C:/Users/jvtor/Downloads/SUPERCARS IN SÃO PAULO! - 812, Turbo S, GT3 RS..mp4")
ret, frame = cap.read()

if ret:
    detector = DetectorPlacaYOLO()
    pre_processador = PreProcessadorROI()
    leitor = LeitorOCR()
    validador = ValidadorPlaca()

    # Detecta placas
    rois = detector.detectar(frame)
    print(f"Placas detectadas: {len(rois)}")

    for i, (x1, y1, x2, y2) in enumerate(rois):
        print(f"Placa {i + 1}: {x1},{y1} - {x2},{y2}")

        # Recorta
        placa = frame[y1:y2, x1:x2]
        cv2.imwrite(f"placa_{i}_original.jpg", placa)

        # Processa
        placa_processada = pre_processador.processar(placa)
        cv2.imwrite(f"placa_{i}_processada.jpg", placa_processada)

        # OCR
        texto = leitor.ler_placa(placa_processada)
        print(f"OCR: {texto}")

        # Valida
        validado = validador.corrigir_e_validar(texto)
        print(f"Validado: {validado}")

cap.release()