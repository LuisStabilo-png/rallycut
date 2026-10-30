"""Detecção da bola de vôlei com o Faster R-CNN do COCO (classe "sports ball"), sem treino próprio."""

import numpy as np

from rallycut.pessoas import carregar_detector, detectar_pessoas

BOLA_COCO = 37


def carregar_detector_bola(altura: int, device: str = "cuda", limiar: float = 0.05):
    """Mesmo detector das pessoas, com limiar baixo (a bola é pequena e tem score baixo) e a resolução
    interna fixada em `altura` (o torchvision reescalaria o frame para 800 px por conta própria)."""
    modelo = carregar_detector(device)
    modelo.roi_heads.score_thresh = limiar
    modelo.transform.min_size, modelo.transform.max_size = (altura,), round(altura * 16 / 9) + 8
    return modelo


def detectar_bolas(modelo, imagens_bgr: list[np.ndarray], device: str = "cuda") -> list[np.ndarray]:
    """Caixas de bola por imagem: (n, 5) com x1, y1, x2, y2 normalizados (0-1) e score."""
    return detectar_pessoas(modelo, imagens_bgr, device, classe=BOLA_COCO)
