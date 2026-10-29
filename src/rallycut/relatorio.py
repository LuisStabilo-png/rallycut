"""Imagens de conferência visual do alinhamento bruto/editado."""

from pathlib import Path

import cv2
import numpy as np


def rotular(imagem: np.ndarray, texto: str) -> np.ndarray:
    """Escreve `texto` no canto superior esquerdo, com fundo escuro para leitura."""
    img = imagem.copy()
    (w, h), _ = cv2.getTextSize(texto, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    cv2.rectangle(img, (0, 0), (w + 8, h + 10), (0, 0, 0), -1)
    cv2.putText(img, texto, (4, h + 3), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return img


def amostra_alinhamento(trechos, bruto: Path, editado: Path, destino: Path, n: int = 8,
                        largura: int = 480) -> None:
    """Frame do editado (esq.) e do bruto (dir.) no meio de n trechos sorteados.

    Busca por TEMPO (não por índice): o bruto de celular tem fps variável, e o índice
    sairia errado ao longo do vídeo."""
    def frame(video, t):
        cap = cv2.VideoCapture(str(video))
        cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
        ok, img = cap.read()
        cap.release()
        return cv2.resize(img, (largura, round(img.shape[0] * largura / img.shape[1]))) if ok else None

    linhas = []
    for r in trechos.sample(min(n, len(trechos)), random_state=0).sort_index().itertuples():
        meio = (r.editado_fim - r.editado_ini) / 2
        a, b = frame(editado, r.editado_ini + meio), frame(bruto, r.bruto_ini + meio)
        if a is not None and b is not None:
            linhas.append(rotular(np.hstack([a, b]), f"trecho {r.rally}  editado | bruto"))
    cv2.imwrite(str(destino), np.vstack(linhas), [cv2.IMWRITE_JPEG_QUALITY, 80])
