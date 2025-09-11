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
    CORRIGIDO: Agora lida com thumbnails de diferentes tamanhos
    """
    altura_frame, largura_frame, _ = frame.shape

    # Cria a área da barra
    barra = np.zeros((altura_barra, largura_frame, 3), dtype=np.uint8)
    barra[:] = (40, 40, 40)  # Cor cinza escuro

    cv2.putText(barra, "Placas Capturadas:", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    x_inicial = 220
    espacamento = 10

    # Desenha cada thumbnail na barra
    for captura in capturas:
        placa_texto = captura["texto"]
        thumbnail = captura["thumbnail"]

        # CORREÇÃO: Redimensiona mantendo a proporção e preenche com fundo
        th_h, th_w, _ = thumbnail.shape

        # Calcula a proporção para altura fixa
        altura_alvo = altura_barra - 20
        proporcao = altura_alvo / th_h
        novo_w = int(th_w * proporcao)
        novo_h = altura_alvo

        # Redimensiona o thumbnail
        thumbnail_redimensionado = cv2.resize(thumbnail, (novo_w, novo_h))

        # CORREÇÃO: Cria uma imagem de fundo com tamanho fixo
        largura_maxima = 120  # Largura máxima para cada thumbnail
        thumbnail_final = np.zeros((novo_h, largura_maxima, 3), dtype=np.uint8)
        thumbnail_final[:] = (40, 40, 40)  # Fundo cinza

        # Centraliza o thumbnail na área disponível
        if novo_w <= largura_maxima:
            x_offset = (largura_maxima - novo_w) // 2
            thumbnail_final[0:novo_h, x_offset:x_offset + novo_w] = thumbnail_redimensionado
        else:
            # Se for muito largo, redimensiona novamente
            proporcao_largura = largura_maxima / novo_w
            novo_w_final = largura_maxima
            novo_h_final = int(novo_h * proporcao_largura)
            thumbnail_redimensionado = cv2.resize(thumbnail, (novo_w_final, novo_h_final))
            y_offset = (novo_h - novo_h_final) // 2
            thumbnail_final[y_offset:y_offset + novo_h_final, 0:novo_w_final] = thumbnail_redimensionado

        # Adiciona o thumbnail à barra
        if x_inicial + largura_maxima <= largura_frame:
            barra[10:10 + novo_h, x_inicial:x_inicial + largura_maxima] = thumbnail_final

            # Adiciona o texto da placa abaixo do thumbnail
            cv2.putText(barra, placa_texto, (x_inicial, altura_barra - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

            x_inicial += largura_maxima + espacamento
        else:
            break  # Não cabe mais thumbnails

    # Combina a barra com o frame principal
    frame[altura_frame - altura_barra:altura_frame, 0:largura_frame] = barra