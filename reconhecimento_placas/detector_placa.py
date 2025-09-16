# detector_placa.py
import cv2
from ultralytics import YOLO

class DetectorPlacaYOLO:
    """
    Classe responsável pela detecção de placas veiculares utilizando o modelo YOLO
    treinado especificamente para essa tarefa.

    A detecção é direta e precisa, eliminando a necessidade de heurísticas
    para encontrar a placa dentro da Região de Interesse (ROI) do carro.
    """

    def __init__(self, model_path='C:/Users/jvtor/Documents/ALPR/runs/detect/train3/weights/best.pt'):
        """
        Inicializa o detector com o modelo YOLO customizado para placas.
        """
        # Carrega o modelo que foi treinado por você
        self.model = YOLO(model_path)
        self.model.to("cuda")  # força GPU

        # O modelo treinado possui uma única classe: 'license_plate', com ID 0.

    def detectar(self, frame):
        """
        Detecta placas veiculares no frame fornecido.

        Args:
            frame: O frame do vídeo a ser processado.

        Returns:
            Uma lista de tuplas, onde cada tupla contém as coordenadas (x1, y1, x2, y2)
            das placas detectadas.
        """
        # Executa a detecção no frame. O modelo já foi treinado para encontrar placas.
        # Não precisamos mais buscar a classe 'car'.
        results = self.model(frame)
        rois = []

        # Para cada objeto detectado no frame, ele já será uma placa
        for result in results:
            boxes = result.boxes
            # A classe ID para o seu modelo treinado (com uma classe) será sempre 0
            if len(boxes) > 0:
                for box in boxes:
                    # Apenas pegamos as coordenadas da caixa delimitadora
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    # Adiciona as coordenadas da placa à lista de ROIs
                    rois.append((x1, y1, x2, y2))

        return rois

# --- Exemplo de uso da classe ---
if __name__ == "__main__":
    # Substitua este caminho pelo caminho real do seu modelo treinado
    detector = DetectorPlacaYOLO()

    # Exemplo com uma imagem estática para demonstração
    # Certifique-se de ter uma imagem de teste
    image_path = "caminho/para/sua/imagem_teste.jpg"
    frame = cv2.imread(image_path)

    if frame is not None:
        rois_detectadas = detector.detectar(frame)

        # Desenha os retângulos nas ROIs detectadas
        for x1, y1, x2, y2 in rois_detectadas:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Mostra a imagem resultante
        cv2.imshow('Detecao de Placas', frame)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print("Erro: Não foi possível carregar a imagem. Verifique o caminho.")