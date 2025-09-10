"""
Treinamento de modelo YOLO para Reconhecimento de Placas Veiculares
Autor: Elias Teófilo
Descrição: Este script implementa o treinamento de um modelo YOLOv8 
para detecção de placas veiculares, conforme fundamentação teórica do TCC.

VERSÃO OTIMIZADA PARA HARDWARE LIMITADO:
- Modelo base alterado para yolov8n.pt para treinamento rápido e eficiente.
- Parâmetros de treino ajustados para priorizar velocidade e evitar erros de memória.
- Adicionado Early Stopping para evitar overfitting e otimizar o tempo de treino.
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
    if img is None:
        print(f"Erro: Não foi possível carregar a imagem em {image_path}")
        return None
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    equalized = cv2.equalizeHist(blur)
    thresh = cv2.adaptiveThreshold(equalized, 255, 
                                   cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY, 11, 2)
    kernel = np.ones((3,3), np.uint8)
    morph = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    # YOLO espera 3 canais, então convertemos a imagem P&B de volta para o formato BGR
    processed_bgr = cv2.cvtColor(morph, cv2.COLOR_GRAY2BGR)

    if save_path:
        cv2.imwrite(save_path, processed_bgr)
    return processed_bgr

# ==========================
# Treinamento do Modelo YOLO
# ==========================
def train_model(class_name, data_yaml="data.yaml"):
    """
    Treina o modelo YOLOv8 para detecção de placas veiculares.
    Salva métricas em CSV e gera gráficos.
    """

    # --- PARÂMETROS OTIMIZADOS PARA VELOCIDADE E HARDWARE ---
    MODEL_VARIANT = 'yolov8n.pt'  # <-- ALTERADO: Modelo nano para treinamento rápido.
    IMG_SIZE = 320                # <-- ALTERADO: Tamanho de imagem reduzido para treinar mais rápido.
    MAX_EPOCHS = 100              # Mantido em 100 para ter um número razoável de épocas.
    PATIENCE = 10                 # <-- ALTERADO: Early Stopping mais agressivo para treino rápido.
    BATCH_SIZE = 8                # <-- ALTERADO: Batch size fixo para ser seguro na 1050 2GB.
    OPTIMIZER = 'AdamW'           # Mantido, é uma boa escolha.
    PROJECT_NAME = 'runs/placas_otimizado_rapido' # Novo nome para o diretório de resultados.

    model = YOLO(MODEL_VARIANT)

    print(f"Iniciando treinamento com o modelo {MODEL_VARIANT}...")

    # Treinamento com parâmetros avançados
    results = model.train(
        data=data_yaml,
        epochs=MAX_EPOCHS,
        patience=PATIENCE,
        batch=BATCH_SIZE,
        imgsz=IMG_SIZE,
        optimizer=OPTIMIZER,
        project=PROJECT_NAME,
        name=f'{class_name.replace(" ", "_")}_treino',

        # --- Parâmetros de Data Augmentation ---
        # Mantidos, pois ajudam na generalização sem um alto custo de processamento.
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=10.0,
        translate=0.1,
        scale=0.5,
        shear=2.0,
        perspective=0.0,
        flipud=0.0,
        fliplr=0.5,
        mosaic=0.5,           # <-- ALTERADO: Mosaic para 0.5. Pode ser mais lento com hardware limitado.
        mixup=0.0             # <-- ALTERADO: Desabilitado para um treino mais rápido.
    )
    
    # Extrai o DataFrame de métricas diretamente dos resultados
    metrics_df = results.csv_to_df()

    # Salva em CSV no diretório da execução
    save_dir = results.save_dir
    csv_path = os.path.join(save_dir, "metricas_finais.csv")
    metrics_df.to_csv(csv_path, index=False)
    print(f"Métricas salvas em: {csv_path}")

    print(f"Gráficos de treinamento e resultados salvos em: {save_dir}")

if __name__ == "__main__":
    # Exemplo de uso para o projeto de placas de carro
    # Certifique-se de que o 'data.yaml' aponta para seu dataset
    train_model("placa de carro", data_yaml="data.yaml")