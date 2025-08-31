"""
Treinamento de modelo YOLO para Reconhecimento de Placas Veiculares
Autor: Elias Teófilo
Descrição: Este script implementa o treinamento de um modelo YOLOv8 
para detecção de placas veiculares, conforme fundamentação teórica do TCC:
- Uso de Redes Neurais Convolucionais (CNNs)
- Detecção em tempo real (YOLO)
- Pré-processamento de imagens (PDI)
- Extração de métricas para avaliação (box_loss, mAP, precisão, recall, FPS)
"""

import argparse
from ultralytics import YOLO
import pandas as pd
import matplotlib.pyplot as plt
import os
import cv2
import numpy as np
import time

# ==========================
# Funções de Pré-processamento (PDI)
# ==========================
def preprocess_image(image_path, save_path=None):
    """
    Aplica pré-processamento em uma imagem, conforme descrito no TCC:
    - Conversão para escala de cinza
    - Remoção de ruído (GaussianBlur)
    - Realce de contraste (Equalização de Histograma)
    - Binarização (Threshold adaptativo)
    - Operações morfológicas (Dilatação + Erosão)
    """
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    equalized = cv2.equalizeHist(blur)
    thresh = cv2.adaptiveThreshold(equalized, 255, 
                                   cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY, 11, 2)
    kernel = np.ones((3,3), np.uint8)
    morph = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    if save_path:
        cv2.imwrite(save_path, morph)
    return morph

# ==========================
# Treinamento do Modelo YOLO
# ==========================
def train_model(class_name, data_yaml="data.yaml", epochs=100, imgsz=320):
    """
    Treina o modelo YOLOv8 para detecção de placas veiculares.
    Salva métricas em CSV e gera gráficos.
    """

    # Conforme discutido no TCC, utilizamos YOLO para detecção em tempo real
    model = YOLO("yolov8n.pt")

    # Treinamento
    results = model.train(data=data_yaml, epochs=epochs, imgsz=imgsz)

    # Extração de métricas
    metrics_list = []
    for epoch in range(epochs):
        metrics = {
            "epoch": epoch + 1,
            "class": class_name,
            "box_loss": results.box.loss[epoch] if hasattr(results.box, "loss") and len(results.box.loss) > epoch else None,
            "mAP50": results.box.map50 if hasattr(results.box, "map50") else None,
            "mAP50-95": results.box.map50_95[epoch] if hasattr(results.box, "map50_95") and len(results.box.map50_95) > epoch else None,
            "precision": results.box.p[0] if hasattr(results.box, "p") else None,
            "recall": results.box.r[0] if hasattr(results.box, "r") else None,
            "fps": results.speed["inference"] if hasattr(results, "speed") else None,
        }
        metrics_list.append(metrics)

    # Converte para DataFrame
    metrics_df = pd.DataFrame(metrics_list)

    # Salva em CSV (acrescentando se já existir)
    csv_path = "C:\\Users\\Elias\\projeto-placaas\\ALPR\\data_placasmetrics.csv"
    if os.path.exists(csv_path):
        existing_df = pd.read_csv(csv_path)
        metrics_df = pd.concat([existing_df, metrics_df], ignore_index=True)
    metrics_df.to_csv(csv_path, index=False)

    # Gráficos
    plt.figure(figsize=(12, 6))
    plt.plot(metrics_df["epoch"], metrics_df["box_loss"], marker="o", label="Box Loss")
    plt.title("Perda da Caixa Delimitadora por Época - Detecção de Placas")
    plt.xlabel("Época"); plt.ylabel("Box Loss")
    plt.xticks(metrics_df["epoch"]); plt.grid(); plt.legend(); plt.show()

    plt.figure(figsize=(12, 6))
    plt.plot(metrics_df["epoch"], metrics_df["mAP50"], marker="o", label="mAP50", color="orange")
    plt.title("mAP50 por Época - Detecção de Placas")
    plt.xlabel("Época"); plt.ylabel("mAP50")
    plt.xticks(metrics_df["epoch"]); plt.grid(); plt.legend(); plt.show()

    plt.figure(figsize=(12, 6))
    plt.plot(metrics_df["epoch"], metrics_df["mAP50-95"], marker="o", label="mAP50-95", color="green")
    plt.title("mAP50-95 por Época - Detecção de Placas")
    plt.xlabel("Época"); plt.ylabel("mAP50-95")
    plt.xticks(metrics_df["epoch"]); plt.grid(); plt.legend(); plt.show()

if __name__ == "__main__":
    # Exemplo de uso para o projeto de placas de carro
    train_model("placa de carro", data_yaml="data.yaml", epochs=100, imgsz=320)
