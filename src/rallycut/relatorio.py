"""Miniaturas antes/depois dos cortes para validação visual."""

from pathlib import Path

import cv2
import numpy as np


def par_de_miniaturas(video: Path | str, frame: int, largura: int = 480) -> np.ndarray:
    """Imagem lado a lado: último frame antes do corte (esquerda) e primeiro depois (direita).

    `frame` é o índice do primeiro frame após o corte (coluna `frame` de cortes.csv)."""
    cap = cv2.VideoCapture(str(video))
    try:
        cap.set(cv2.CAP_PROP_POS_FRAMES, max(frame - 1, 0))
        ok_a, antes = cap.read()
        ok_d, depois = cap.read()
    finally:
        cap.release()
    if not (ok_a and ok_d):
        raise RuntimeError(f"não consegui ler os frames {frame - 1}/{frame} de {video}")
    altura = round(antes.shape[0] * largura / antes.shape[1])
    antes, depois = (cv2.resize(f, (largura, altura), interpolation=cv2.INTER_AREA)
                     for f in (antes, depois))
    return np.hstack([antes, np.full((altura, 4, 3), 255, np.uint8), depois])


def rotular(imagem: np.ndarray, texto: str) -> np.ndarray:
    """Escreve `texto` no canto superior esquerdo, com fundo escuro para leitura."""
    img = imagem.copy()
    (w, h), _ = cv2.getTextSize(texto, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    cv2.rectangle(img, (0, 0), (w + 8, h + 10), (0, 0, 0), -1)
    cv2.putText(img, texto, (4, h + 3), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return img
