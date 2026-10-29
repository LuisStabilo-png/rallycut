"""Detecção de pessoas (torchvision) e continuidade dos jogadores entre dois frames.

Num corte real os jogadores "teletransportam"; no jogo contínuo cada um se move pouco
entre frames vizinhos. Os jogadores mais próximos da câmera (caixas maiores) aparecem
quase sempre inteiros no quadro, então são a âncora mais confiável para esse teste.
"""

from pathlib import Path

import cv2
import numpy as np
import torch
from scipy.optimize import linear_sum_assignment

PASTA_MODELOS = Path(__file__).resolve().parents[2] / "data" / "modelos"
PESSOA_COCO = 1


def carregar_detector(device: str = "cuda"):
    """Faster R-CNN MobileNetV3 (COCO). Os pesos ficam em data/modelos/, fora do git."""
    from torchvision.models.detection import (FasterRCNN_MobileNet_V3_Large_FPN_Weights,
                                              fasterrcnn_mobilenet_v3_large_fpn)
    PASTA_MODELOS.mkdir(parents=True, exist_ok=True)
    torch.hub.set_dir(str(PASTA_MODELOS))
    modelo = fasterrcnn_mobilenet_v3_large_fpn(
        weights=FasterRCNN_MobileNet_V3_Large_FPN_Weights.DEFAULT, box_score_thresh=0.4)
    return modelo.eval().to(device)


def ler_par_frames(video: Path | str, frame: int) -> tuple[np.ndarray, np.ndarray]:
    """Último frame antes e primeiro depois do corte (BGR, resolução original)."""
    cap = cv2.VideoCapture(str(video))
    try:
        cap.set(cv2.CAP_PROP_POS_FRAMES, max(frame - 1, 0))
        ok_a, antes = cap.read()
        ok_d, depois = cap.read()
    finally:
        cap.release()
    if not (ok_a and ok_d):
        raise RuntimeError(f"não consegui ler os frames {frame - 1}/{frame} de {video}")
    return antes, depois


def extrair_pares(video: Path | str, frames: list[int], largura: int = 640, altura: int = 360,
                  nvdec: bool = True):
    """Gera (frame, antes, depois) para cada corte, numa única passada pelo vídeo.

    `frame` é o primeiro frame após o corte; `antes` é o frame anterior. Decodificar o
    vídeo em sequência (NVDEC) é muito mais rápido do que buscar cada par com seek."""
    import subprocess
    import tempfile

    alvo = set(int(f) for f in frames)
    pre = ["-hwaccel", "cuda", "-hwaccel_output_format", "cuda"] if nvdec else []
    filtro = (f"scale_cuda={largura}:{altura},hwdownload,format=nv12" if nvdec
              else f"scale={largura}:{altura}")
    cmd = ["ffmpeg", "-nostdin", "-loglevel", "error", *pre, "-i", str(video), "-an",
           "-fps_mode", "passthrough", "-vf", filtro, "-pix_fmt", "bgr24", "-f", "rawvideo", "-"]
    tam = largura * altura * 3
    anterior = None
    with tempfile.TemporaryFile() as erros:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=erros)
        i = 0
        while len(alvo) and (dados := proc.stdout.read(tam)) and len(dados) == tam:
            atual = np.frombuffer(dados, np.uint8).reshape(altura, largura, 3)
            if i in alvo:
                yield i, anterior, atual.copy()
                alvo.discard(i)
            anterior = atual.copy() if (i + 1) in alvo else None
            i += 1
        proc.kill()
        proc.wait()


@torch.no_grad()
def detectar_pessoas(modelo, imagens_bgr: list[np.ndarray], device: str = "cuda") -> list[np.ndarray]:
    """Caixas de pessoas por imagem: array (n, 5) com x1, y1, x2, y2 normalizados (0-1) e score."""
    tensores = [torch.from_numpy(cv2.cvtColor(i, cv2.COLOR_BGR2RGB)).permute(2, 0, 1).float().div(255).to(device)
                for i in imagens_bgr]
    saidas = modelo(tensores)
    resultado = []
    for img, s in zip(imagens_bgr, saidas):
        h, w = img.shape[:2]
        m = (s["labels"] == PESSOA_COCO).cpu().numpy()
        caixas = s["boxes"].cpu().numpy()[m] / np.array([w, h, w, h])
        resultado.append(np.hstack([caixas, s["scores"].cpu().numpy()[m, None]]) if m.any()
                         else np.zeros((0, 5)))
    return resultado


def verificar_cortes(video: Path | str, frames: list[int], modelo, device: str = "cuda",
                     lote: int = 8, progresso=print) -> "pd.DataFrame":
    """Métricas de continuidade dos jogadores para cada corte candidato (coluna `frame`)."""
    import pandas as pd

    linhas, fila = [], []

    def processar():
        imgs = [x for _, a, d in fila for x in (a, d)]
        deteccoes = detectar_pessoas(modelo, imgs, device)
        for (f, _, _), a, b in zip(fila, deteccoes[0::2], deteccoes[1::2]):
            linhas.append({"frame": f, **continuidade(a, b)})
        fila.clear()

    for f, antes, depois in extrair_pares(video, frames):
        if antes is None:
            continue
        fila.append((f, antes, depois))
        if len(fila) >= lote:
            processar()
            if progresso and len(linhas) % 200 < lote:
                progresso(f"  jogadores: {len(linhas)}/{len(frames)}")
    if fila:
        processar()
    return pd.DataFrame(linhas)


def _pes_e_altura(caixas: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    pes = np.stack([(caixas[:, 0] + caixas[:, 2]) / 2, caixas[:, 3]], axis=1)
    return pes, np.maximum(caixas[:, 3] - caixas[:, 1], 1e-3)


def continuidade(antes: np.ndarray, depois: np.ndarray, frac_proximos: float = 0.6) -> dict:
    """Mede o quanto os jogadores de `antes` reaparecem no mesmo lugar em `depois`.

    Distâncias são entre os pés (centro da base da caixa), divididas pela altura da pessoa,
    para que um jogador próximo da câmera e um distante sejam comparáveis. Valores altos
    indicam troca de cena. "Próximos" são as pessoas com altura >= frac_proximos * a maior."""
    vazio = {"n_antes": len(antes), "n_depois": len(depois), "perto_max": np.nan, "perto_med": np.nan,
             "maior": np.nan, "frac_estavel": np.nan, "custo_hung": np.nan}
    if len(antes) == 0 or len(depois) == 0:
        return vazio
    pa, ha = _pes_e_altura(antes)
    pb, _ = _pes_e_altura(depois)
    dist = np.linalg.norm(pa[:, None, :] - pb[None, :, :], axis=2) / ha[:, None]
    mais_proxima = dist.min(axis=1)  # por pessoa de `antes`, distância à mais próxima de `depois`
    perto = ha >= frac_proximos * ha.max()
    linhas, colunas = linear_sum_assignment(np.minimum(dist, 3.0))
    return {**vazio,
            "perto_max": float(mais_proxima[perto].max()),
            "perto_med": float(np.median(mais_proxima[perto])),
            "maior": float(mais_proxima[ha.argmax()]),
            "frac_estavel": float((mais_proxima < 0.3).mean()),
            "custo_hung": float(np.minimum(dist, 3.0)[linhas, colunas].mean())}
