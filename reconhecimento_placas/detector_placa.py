# detector_placa.py
import cv2
from ultralytics import YOLO

class DetectorPlacaYOLO:
    """
    Classe responsável pela detecção de placas veiculares utilizando o modelo YOLO.
    Corresponde à etapa de "Detecção de Objetos com YOLO" (Seção 2.4 do TCC).
    O YOLO utiliza Redes Neurais Convolucionais (CNNs) para realizar a detecção
    de forma rápida e precisa, ideal para aplicações em tempo real[cite: 167, 169].
    """
    def __init__(self, model_path='yolov8n.pt'):
        """
        Inicializa o detector com um modelo YOLO pré-treinado.
        Embora 'yolov8n.pt' seja um modelo geral, ele é eficaz para encontrar
        veículos, e um modelo treinado especificamente para placas (custom model)
        seria ainda mais preciso. Para este exemplo, usaremos um modelo que detecta carros
        e, em seguida, buscaremos a placa na ROI do carro. Uma abordagem mais direta
        seria usar um modelo já treinado para 'license_plate'.
        """
        self.model = YOLO(model_path)

    def detectar(self, frame):
        """
        Detecta placas no frame fornecido.

        Args:
            frame: O frame do vídeo a ser processado.

        Returns:
            Uma lista de tuplas, onde cada tupla contém as coordenadas (x1, y1, x2, y2)
            da Região de Interesse (ROI) onde uma placa foi detectada.
        """
        # A detecção é tratada como um problema de regressão unificada,
        # o que torna o YOLO extremamente rápido[cite: 174].
        results = self.model(frame)
        rois = []

        # Para cada objeto detectado no frame
        for result in results:
            boxes = result.boxes
            for box in boxes:
                # Obtenha as coordenadas do bounding box
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                
                # Para um modelo genérico, filtramos por classe (ex: 'car').
                # Se tivéssemos um modelo treinado para 'license_plate',
                # usaríamos o ID dessa classe.
                class_id = int(box.cls[0])
                class_name = self.model.names[class_id]

                # Simulação: assumindo que a placa está na parte inferior central do carro.
                # A abordagem ideal é usar um detector de placas treinado.
                if class_name == 'car':
                    # Heurística para encontrar a placa dentro da ROI do carro
                    car_h = y2 - y1
                    car_w = x2 - x1
                    # Estimativa da ROI da placa
                    plate_y1 = y1 + int(car_h * 0.7)
                    plate_y2 = y2 - int(car_h * 0.1)
                    plate_x1 = x1 + int(car_w * 0.3)
                    plate_x2 = x2 - int(car_w * 0.3)
                    
                    rois.append((plate_x1, plate_y1, plate_x2, plate_y2))
        
        # Em um cenário real com um modelo treinado para placas:
        # if class_name == 'license_plate':
        #     rois.append((x1, y1, x2, y2))

        return rois