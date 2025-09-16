# utils.py
import cv2
import numpy as np

def desenhar_resultados(frame, roi_ajustada, texto_placa):
    """
    Desenha o bounding box e o texto da placa reconhecida no frame.
    - Verde se validada
    - Vermelho apenas box se não validada (sem texto)
    """
    x1, y1, x2, y2 = roi_ajustada

    if texto_placa:  # Placa válida
        cor_box = (0, 255, 0)
        cv2.rectangle(frame, (x1, y1), (x2, y2), cor_box, 2)

        texto_exibicao = f"Placa: {texto_placa}"
        (largura_texto, altura_texto), _ = cv2.getTextSize(
            texto_exibicao, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)

        pos_texto_y = y1 - 10 if y1 - 10 > 10 else y1 + altura_texto + 20

        cv2.rectangle(frame, (x1, pos_texto_y - altura_texto - 5),
                      (x1 + largura_texto, pos_texto_y + 5), (0, 0, 0), -1)

        cv2.putText(frame, texto_exibicao, (x1, pos_texto_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, cor_box, 2)
    else:  # Não validada → só retângulo vermelho
        cor_box = (0, 0, 255)
        cv2.rectangle(frame, (x1, y1), (x2, y2), cor_box, 2)


def desenhar_barra_capturas(frame, capturas, altura_barra=80):
    """
    Desenha uma barra na parte inferior do frame com os thumbnails das placas capturadas.
    """
    altura_frame, largura_frame, _ = frame.shape

    barra = np.zeros((altura_barra, largura_frame, 3), dtype=np.uint8)
    barra[:] = (40, 40, 40)  # fundo cinza escuro

    cv2.putText(barra, "Placas Capturadas:", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    x_inicial = 220
    espacamento = 10

    for captura in capturas:
        placa_texto = captura["texto"]
        thumbnail = captura["thumbnail"]

        th_h, th_w, _ = thumbnail.shape
        altura_alvo = altura_barra - 20
        proporcao = altura_alvo / th_h
        novo_w = int(th_w * proporcao)
        novo_h = altura_alvo

        thumbnail_redimensionado = cv2.resize(thumbnail, (novo_w, novo_h))

        largura_maxima = 120
        thumbnail_final = np.zeros((novo_h, largura_maxima, 3), dtype=np.uint8)
        thumbnail_final[:] = (40, 40, 40)

        if novo_w <= largura_maxima:
            x_offset = (largura_maxima - novo_w) // 2
            thumbnail_final[0:novo_h, x_offset:x_offset + novo_w] = thumbnail_redimensionado
        else:
            proporcao_largura = largura_maxima / novo_w
            novo_w_final = largura_maxima
            novo_h_final = int(novo_h * proporcao_largura)
            thumbnail_redimensionado = cv2.resize(thumbnail, (novo_w_final, novo_h_final))
            y_offset = (novo_h - novo_h_final) // 2
            thumbnail_final[y_offset:y_offset + novo_h_final, 0:novo_w_final] = thumbnail_redimensionado

        if x_inicial + largura_maxima <= largura_frame:
            barra[10:10 + novo_h, x_inicial:x_inicial + largura_maxima] = thumbnail_final

            cv2.putText(barra, placa_texto, (x_inicial, altura_barra - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

            x_inicial += largura_maxima + espacamento
        else:
            break

    frame[altura_frame - altura_barra:altura_frame, 0:largura_frame] = barra

def desenhar_barra_detectadas(frame, detectadas, altura_barra=50):
    """
    Barra superior mostrando placas detectadas pelo YOLO (tempo real).
    """
    altura_frame, largura_frame, _ = frame.shape

    barra = np.zeros((altura_barra, largura_frame, 3), dtype=np.uint8)
    barra[:] = (60, 60, 60)  # cinza escuro

    cv2.putText(barra, "Placas Detectadas (YOLO):", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    x_inicial = 280
    espacamento = 120

    for i, roi_coords in enumerate(detectadas):
        label = f"Placa {i+1}"
        cv2.putText(barra, label, (x_inicial, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        x_inicial += espacamento
        if x_inicial > largura_frame - 100:
            break

    # cola a barra no topo
    frame[0:altura_barra, 0:largura_frame] = barra