# utils.py
import cv2
import numpy as np

def desenhar_resultados(frame, roi_ajustada, texto_placa):
    """
    Desenha o bounding box e o texto da placa reconhecida no frame.
    """
    x1, y1, x2, y2 = roi_ajustada
    
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    if texto_placa:
        texto_exibicao = f"Placa: {texto_placa}"
        cor_texto = (0, 255, 0)
    else:
        texto_exibicao = "nao verificada"
        cor_texto = (0, 0, 255)

    (largura_texto, altura_texto), _ = cv2.getTextSize(
        texto_exibicao, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
    
    pos_texto_y = y1 - 10 if y1 - 10 > 10 else y1 + altura_texto + 20
    
    cv2.rectangle(frame, (x1, pos_texto_y - altura_texto - 5), 
                  (x1 + largura_texto, pos_texto_y + 5), (0, 0, 0), -1)
    
    cv2.putText(frame, texto_exibicao, (x1, pos_texto_y), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, cor_texto, 2)

def desenhar_barra_capturas(frame, capturas, altura_barra=80):
    """
    Desenha uma barra na parte inferior do frame com os thumbnails das placas capturadas.
    """
    altura_frame, largura_frame, _ = frame.shape
    
    # Cria a área da barra
    barra = np.zeros((altura_barra, largura_frame, 3), dtype=np.uint8)
    barra[:] = (40, 40, 40) # Cor cinza escuro

    cv2.putText(barra, "Placas Capturadas:", (10, 25), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    x_inicial = 220
    espacamento = 10
    
    # Desenha cada thumbnail na barra
    for captura in capturas:
        placa_texto = captura["texto"]
        thumbnail = captura["thumbnail"]
        
        # Redimensiona o thumbnail para caber na barra
        th_h, th_w, _ = thumbnail.shape
        ratio = (altura_barra - 20) / th_h
        novo_w, novo_h = int(th_w * ratio), int(th_h * ratio)
        thumbnail_redimensionado = cv2.resize(thumbnail, (novo_w, novo_h))
        
        # Adiciona o thumbnail à barra
        barra[10:10 + novo_h, x_inicial:x_inicial + novo_w] = thumbnail_redimensionado
        
        # Adiciona o texto da placa abaixo do thumbnail
        cv2.putText(barra, placa_texto, (x_inicial, altura_barra - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        x_inicial += novo_w + espacamento

    # Combina a barra com o frame principal
    frame[altura_frame - altura_barra:altura_frame, 0:largura_frame] = barra