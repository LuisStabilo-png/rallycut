"""Features por janela de 0,5 s de um vídeo bruto: movimento, jogadores e áudio.

Uma única passada pelo vídeo a 4 fps (2 frames por janela): o movimento vem da diferença
entre frames vizinhos e os jogadores são detectados em 1 frame por janela.
"""

import cv2
import numpy as np
import pandas as pd
import soundfile as sf
import torch
from scipy.signal import stft
from torchvision.models import MobileNet_V3_Large_Weights, mobilenet_v3_large

from rallycut.media import iterar_frames
from rallycut.pessoas import PASTA_MODELOS, continuidade, detectar_pessoas

JANELA_S, FPS = 0.5, 4
POR_JANELA = int(JANELA_S * FPS)


def carregar_embedder(device: str = "cuda"):
    """MobileNetV3-Large (ImageNet): imagem -> vetor de 960 números. Pesos em data/modelos/."""
    torch.hub.set_dir(str(PASTA_MODELOS))
    m = mobilenet_v3_large(weights=MobileNet_V3_Large_Weights.DEFAULT)
    return torch.nn.Sequential(m.features, m.avgpool, torch.nn.Flatten()).eval().to(device)


@torch.no_grad()
def embeddings(modelo, frames_bgr: list, device: str = "cuda") -> np.ndarray:
    x = torch.from_numpy(np.stack(frames_bgr)[..., ::-1].copy()).to(device).permute(0, 3, 1, 2).float() / 255
    media = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    desvio = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
    return modelo((x - media) / desvio).cpu().numpy()


def _audio(wav, n: int) -> pd.DataFrame:
    """Energia (dB), fração na banda do impacto (800-5000 Hz) e fluxo espectral médio por janela."""
    x, sr = sf.read(str(wav), dtype="float32")
    f, t, Z = stft(x, sr, nperseg=int(0.025 * sr), noverlap=int(0.015 * sr), boundary=None, padded=False)
    p = np.abs(Z) ** 2
    banda = p[(f >= 800) & (f <= 5000)]
    fluxo = np.r_[0, np.maximum(np.diff(np.log(banda + 1e-10), axis=1), 0).mean(axis=0)]
    j = np.minimum((t / JANELA_S).astype(int), n - 1)
    conta = np.maximum(np.bincount(j, minlength=n), 1)

    def media(v):
        return np.bincount(j, weights=v, minlength=n) / conta

    total = p.sum(axis=0) + 1e-10
    return pd.DataFrame({"rms_db": 10 * np.log10(media(total) + 1e-10),
                         "banda": media(banda.sum(axis=0) / total), "fluxo": media(fluxo)})


def _jogadores(caixas: list) -> pd.DataFrame:
    """Contagem, tamanho, dispersão e movimento dos jogadores de cada janela (vs a anterior)."""
    linhas = []
    for w, c in enumerate(caixas):
        L = {"n": len(c)}
        if len(c):
            x, y, h = (c[:, 0] + c[:, 2]) / 2, c[:, 3], c[:, 3] - c[:, 1]
            L |= {"h_max": h.max(), "disp_x": x.std(), "disp_y": y.std(), "y_med": y.mean()}
        if w and len(c) and len(caixas[w - 1]):
            m = continuidade(caixas[w - 1], c)
            L |= {k: m[k] for k in ("perto_med", "custo_hung", "frac_estavel")}
        linhas.append(L)
    return pd.DataFrame(linhas, columns=["n", "h_max", "disp_x", "disp_y", "y_med",
                                         "perto_med", "custo_hung", "frac_estavel"])


def sufixo(altura: int) -> str:
    """Sufixo dos arquivos de features por resolução (360 = nomes antigos, sem sufixo)."""
    return "" if altura == 360 else f"_{altura}"


def features_video(video, wav, duracao_s: float, detector, embedder, altura: int = 720,
                   device: str = "cuda", lote: int = 16):
    """(DataFrame, embeddings): uma linha por janela (`t` = início em s, e as features) e a matriz
    (janelas, 960) de embeddings visuais.

    Os frames são decodificados com `altura` px (jogadores pequenos pedem mais pixels); o embedding
    continua calculado a 640x360, para comparar só o efeito da resolução da detecção."""
    n = int(duracao_s / JANELA_S)
    mov = np.full((n * POR_JANELA, 2), np.nan)
    caixas, fila = [np.zeros((0, 5))] * n, []
    emb = np.zeros((n, 960), np.float16)

    def detectar():
        frames = [f for _, f in fila]
        for (w, _), c, e in zip(fila, detectar_pessoas(detector, frames, device),
                                embeddings(embedder, [cv2.resize(f, (640, 360), interpolation=cv2.INTER_AREA)
                                                      for f in frames], device)):
            caixas[w], emb[w] = c, e
        fila.clear()

    anterior = None
    for i, frame in enumerate(iterar_frames(video, FPS, round(altura * 16 / 9 / 2) * 2, altura)):
        if i >= n * POR_JANELA:
            break
        cinza = cv2.cvtColor(cv2.resize(frame, (160, 90), interpolation=cv2.INTER_AREA),
                             cv2.COLOR_BGR2GRAY).astype(np.int16)
        if anterior is not None:
            d = np.abs(cinza - anterior) / 255
            mov[i] = d.mean(), (d > 0.08).mean()
        anterior = cinza
        if i % POR_JANELA == 0:
            fila.append((i // POR_JANELA, frame))
            if len(fila) == lote:
                detectar()
    if fila:
        detectar()

    m = np.nanmean(mov.reshape(n, POR_JANELA, 2), axis=1)
    base = pd.DataFrame({"t": np.arange(n) * JANELA_S, "mov_med": m[:, 0], "mov_frac": m[:, 1]})
    return pd.concat([base, _jogadores(caixas), _audio(wav, n)], axis=1), emb


def rotular_janelas(t: np.ndarray, trechos: pd.DataFrame) -> np.ndarray:
    """1 se o centro da janela cai dentro de algum trecho (rally) do bruto, senão 0."""
    centro = t + JANELA_S / 2
    return np.any([(centro >= a) & (centro <= b) for a, b in zip(trechos["bruto_ini"], trechos["bruto_fim"])],
                  axis=0).astype(int)


def rotular_inicios(t: np.ndarray, trechos: pd.DataFrame, antes_s: float = 3.0, depois_s: float = 2.0) -> np.ndarray:
    """1 se o centro da janela está no entorno do início de algum rally (de `antes_s` antes a `depois_s` depois)."""
    return rotular_janelas(t, pd.DataFrame({"bruto_ini": trechos["bruto_ini"] - antes_s,
                                            "bruto_fim": trechos["bruto_ini"] + depois_s}))
