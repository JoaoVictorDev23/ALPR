"""
Treinamento de modelo YOLO para Reconhecimento de Placas Veiculares
Configurado para RTX 3060 12GB + Ryzen 7 2700
"""

import argparse
from ultralytics import YOLO
import pandas as pd
import matplotlib.pyplot as plt
import os
import cv2
import numpy as np
import time
import torch
import sys

def check_environment():
    """Verifica se o ambiente está configurado corretamente"""
    print("=== VERIFICAÇÃO DO AMBIENTE ===")
    print(f"Python: {sys.version}")
    print(f"CUDA disponível: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

    # Verifica se data.yaml existe
    if os.path.exists("data/data.yaml"):
        print("✓ data.yaml encontrado")
    else:
        print("✗ data.yaml não encontrado - Verifique o caminho")

    # Verifica se os dados de treinamento existem
    if os.path.exists("data_placas/train"):
        train_images = [f for f in os.listdir('data_placas/train') if f.endswith(('.jpg', '.png', '.jpeg'))]
        print(f"✓ {len(train_images)} imagens de treino encontradas")
    else:
        print("✗ Pasta data/train não encontrada")

    # Verifica se o modelo base existe
    if os.path.exists("yolov8n.pt"):
        print("✓ yolov8n.pt encontrado")
    else:
        print("✗ yolov8n.pt não encontrado - Baixando automaticamente...")
        try:
            YOLO("yolov8n.pt")
            print("✓ yolov8n.pt baixado com sucesso")
        except:
            print("✗ Falha ao baixar yolov8n.pt")

    print("===============================")

def train_model(class_name, data_yaml = "data/data.yaml", epochs=100, imgsz=640):
    """
    Treina o modelo YOLOv8 para detecção de placas veiculares.
    Configurado para RTX 3060 12GB.
    """

    # Verifica se CUDA está disponível
    print(f"CUDA disponível: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

    # Carrega modelo YOLOv8n
    model = YOLO("yolov8n.pt")

    # Treinamento otimizado para RTX 3060
    results = model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=imgsz,
        batch=16,           # Batch size otimizado para 12GB VRAM
        device="0",         # Usa GPU 0
        workers=8,          # Otimizado para Ryzen 7 2700 (8 cores)
        patience=50,        # Early stopping
        lr0=0.01,           # Learning rate inicial
        lrf=0.01,           # Learning rate final
        momentum=0.937,
        weight_decay=0.0005,
        warmup_epochs=3.0,
        warmup_momentum=0.8,
        box=7.5,
        cls=0.5,
        dfl=1.5,
        verbose=True
    )

    # Salva o modelo treinado
    os.makedirs("models", exist_ok=True)
    model.save("models/placa_detection.pt")

    # Exporta para formato ONNX (opcional)
    model.export(format="onnx")

    # Salva métricas em CSV
    os.makedirs("data", exist_ok=True)
    metrics_df = pd.DataFrame(results.results_dict)
    metrics_df.to_csv("data/training_metrics.csv", index=False)

    print("Treinamento concluído!")
    print(f"Modelo salvo em: models/placa_detection.pt")
    print(f"Métricas salvas em: data/training_metrics.csv")

if __name__ == "__main__":
    # Primeiro verifica o ambiente
    check_environment()

    # Pergunta se deseja continuar
    resposta = input("\nDeseja continuar com o treinamento? (s/n): ")

    if resposta.lower() == 's':
        print("\nIniciando treinamento...")
        train_model(
            class_name="placa",
            data_yaml="data/data.yaml",
            epochs=100,
            imgsz=640
        )
    else:
        print("Treinamento cancelado.")