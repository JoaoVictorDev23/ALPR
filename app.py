import argparse
from ultralytics import YOLO
import pandas as pd
import matplotlib.pyplot as plt
import os

def train_model(class_name):
    # Carregue o modelo YOLOv8
    model = YOLO('yolov8n.pt')

    # Treine o modelo (custom.yaml já deve estar configurado para suas classes)
    results = model.train(data='data.yaml', epochs=5, imgsz=320)

    # Extraia as informações relevantes após o treinamento
    metrics_list = []
    for epoch in range(len(results.box.p)):
        metrics = {
            'epoch': epoch + 1,
            'class': class_name,  # Adicione a classe aqui
            'box_loss': results.box.loss[epoch] if hasattr(results.box, 'loss') and len(results.box.loss) > epoch else None,
            'mAP50': results.box.map50 if hasattr(results.box, 'map50') and results.box.map50 is not None else None,
            'mAP50-95': results.box.map50_95[epoch] if hasattr(results.box, 'map50_95') and len(results.box.map50_95) > epoch else None,
        }
        metrics_list.append(metrics)

    # Converta as métricas em DataFrame
    metrics_df = pd.DataFrame(metrics_list)
    
    # Caminho para o arquivo CSV único
    csv_path = 'C:\\Users\\Elias\\projeto-placaas\\ALPR\\data_placasmetrics.csv'

    # Verifique se o arquivo já existe para evitar sobrescrever
    if os.path.exists(csv_path):
        existing_df = pd.read_csv(csv_path)
        # Concatenar o novo DataFrame com o existente
        metrics_df = pd.concat([existing_df, metrics_df], ignore_index=True)

    # Escreva (ou atualize) o CSV único
    metrics_df.to_csv(csv_path, index=False)

    # Exibir gráficos dos resultados de treinamento
    plt.figure(figsize=(12, 6))
    plt.plot(metrics_df['epoch'], metrics_df['box_loss'], marker='o', label='Perda da Caixa Delimitadora')
    plt.title(f'Perda da Caixa Delimitadora por Época - Detecção de Placa de Carro')
    plt.xlabel('Época')
    plt.ylabel('Perda da Caixa Delimitadora')
    plt.xticks(metrics_df['epoch'])
    plt.grid()
    plt.legend()
    plt.show()

    plt.figure(figsize=(12, 6))
    plt.plot(metrics_df['epoch'], metrics_df['mAP50'], marker='o', label='mAP50', color='orange')
    plt.title(f'mAP50 por Época - Detecção de Placa de Carro')
    plt.xlabel('Época')
    plt.ylabel('mAP50')
    plt.xticks(metrics_df['epoch'])
    plt.grid()
    plt.legend()
    plt.show()

    plt.figure(figsize=(12, 6))
    plt.plot(metrics_df['epoch'], metrics_df['mAP50-95'], marker='o', label='mAP50-95', color='green')
    plt.title(f'mAP50-95 por Época - Detecção de Placa de Carro')
    plt.xlabel('Época')
    plt.ylabel('mAP50-95')
    plt.xticks(metrics_df['epoch'])
    plt.grid()
    plt.legend()
    plt.show()

if __name__ == "__main__":
    # Exemplo de uso para o projeto de placas de carro
    train_model("placa de carro")
    # Para treinar para outra classe, altere a string
    # train_model("outra classe")