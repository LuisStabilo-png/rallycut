"""Detecção da bola de vôlei: Faster R-CNN do COCO (classe "sports ball"), sem treino, e um mapa de calor próprio."""

import warnings
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models import MobileNet_V3_Large_Weights, mobilenet_v3_large

from rallycut.media import iterar_frames
from rallycut.pessoas import PASTA_MODELOS, carregar_detector, detectar_pessoas

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


# --- detector próprio: mapa de calor da bola (estilo CenterNet), treinado com cliques no centro ---

LARGURA, ALTURA, PASSO = 1280, 720, 8  # entrada do mapa e passo do mapa de calor (160 x 90)
_MEDIA, _DESVIO = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)


class MapaBola(nn.Module):
    """MobileNetV3 (ImageNet) com 3 escalas (passos 8, 16 e 32) fundidas num mapa de calor 1/8 da entrada."""

    def __init__(self):
        super().__init__()
        torch.hub.set_dir(str(PASTA_MODELOS))
        f = mobilenet_v3_large(weights=MobileNet_V3_Large_Weights.DEFAULT).features
        self.b8, self.b16, self.b32 = f[:7], f[7:13], f[13:17]
        self.lat = nn.ModuleList([nn.Conv2d(c, 64, 1) for c in (40, 112, 960)])
        self.cabeca = nn.Sequential(nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(), nn.Conv2d(64, 1, 1))
        self.cabeca[-1].bias.data.fill_(-4.0)

    def forward(self, x):  # (n, 3, ALTURA, LARGURA) normalizado -> logits (n, ALTURA/8, LARGURA/8)
        a = self.b8(x)
        b = self.b16(a)
        c = self.b32(b)
        p = self.lat[0](a) + F.interpolate(self.lat[1](b), a.shape[-2:]) + F.interpolate(self.lat[2](c), a.shape[-2:])
        return self.cabeca(p)[:, 0]

    def train(self, modo: bool = True):
        super().train(modo)
        for m in self.modules():  # lotes pequenos: mantém as estatísticas do ImageNet nas normalizações
            if isinstance(m, nn.BatchNorm2d):
                m.eval()
        return self


def preparar(imagens_rgb: np.ndarray, device: str = "cuda") -> torch.Tensor:
    """(n, ALTURA, LARGURA, 3) uint8 RGB -> tensor normalizado."""
    x = torch.from_numpy(imagens_rgb).to(device).permute(0, 3, 1, 2).float() / 255
    return (x - torch.tensor(_MEDIA, device=device).view(1, 3, 1, 1)) / torch.tensor(_DESVIO, device=device).view(1, 3, 1, 1)


def alvo_gaussiano(centro_px: tuple | None, sigma: float = 1.0) -> np.ndarray:
    """Mapa alvo (ALTURA/8, LARGURA/8): gaussiana (pico 1 na célula do centro); zeros se não há bola.
    `centro_px` na escala da entrada (LARGURA x ALTURA)."""
    h, w = ALTURA // PASSO, LARGURA // PASSO
    alvo = np.zeros((h, w), np.float32)
    if centro_px is None:
        return alvo
    cx, cy = int(centro_px[0] // PASSO), int(centro_px[1] // PASSO)
    if not (0 <= cx < w and 0 <= cy < h):
        return alvo
    yy, xx = np.mgrid[:h, :w]
    return np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * sigma ** 2)).astype(np.float32)


def perda_focal(logits: torch.Tensor, alvo: torch.Tensor) -> torch.Tensor:
    """Focal loss do CenterNet: positivos só no pico (alvo == 1), negativos atenuados perto do pico."""
    p = torch.sigmoid(logits.float()).clamp(1e-4, 1 - 1e-4)
    pos = alvo.eq(1).float()
    positivos = -((1 - p) ** 2) * torch.log(p) * pos
    negativos = -((1 - alvo) ** 4) * p ** 2 * torch.log(1 - p) * (1 - pos)
    return (positivos.sum() + negativos.sum()) / pos.sum().clamp(min=1)


def aumentar(img: np.ndarray, centro: tuple | None, rng) -> tuple[np.ndarray, tuple | None]:
    """Espelho, zoom/deslocamento, troca de canais de cor (a cor da bola muda de jogo para jogo) e brilho."""
    s, tx, ty = rng.uniform(0.8, 1.25), rng.uniform(-0.1, 0.1) * LARGURA, rng.uniform(-0.1, 0.1) * ALTURA
    M = np.array([[s, 0, (1 - s) * LARGURA / 2 + tx], [0, s, (1 - s) * ALTURA / 2 + ty]])
    img = cv2.warpAffine(img, M, (LARGURA, ALTURA), borderMode=cv2.BORDER_REFLECT)
    if centro is not None:
        centro = (centro[0] * s + M[0, 2], centro[1] * s + M[1, 2])
    if rng.random() < 0.5:
        img = img[:, ::-1]
        centro = None if centro is None else (LARGURA - 1 - centro[0], centro[1])
    if rng.random() < 0.5:
        img = img[..., rng.permutation(3)]
    img = np.clip(img.astype(np.float32) * rng.uniform(0.7, 1.3) + rng.uniform(-20, 20), 0, 255).astype(np.uint8)
    return np.ascontiguousarray(img), centro


def treinar_mapa(imagens: np.ndarray, centros: list, passos: int = 1200, lote: int = 4, device: str = "cuda",
                 semente: int = 0) -> MapaBola:
    """Treina o MapaBola. `imagens` (n, ALTURA, LARGURA, 3) RGB; `centros[i]` = (x, y) na escala da
    entrada, ou None (frames sem bola entram como negativos puros)."""
    torch.manual_seed(semente)
    rng = np.random.default_rng(semente)
    modelo = MapaBola().to(device).train()
    otim = torch.optim.AdamW(modelo.parameters(), lr=5e-4, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(otim, max_lr=5e-4, total_steps=passos, pct_start=0.1)
    for _ in range(passos):
        lote_i = rng.integers(0, len(imagens), lote)
        pares = [aumentar(imagens[i], centros[i], rng) for i in lote_i]
        x = preparar(np.stack([p[0] for p in pares]), device)
        alvo = torch.from_numpy(np.stack([alvo_gaussiano(p[1]) for p in pares])).to(device)
        with torch.autocast(device, dtype=torch.float16):
            logits = modelo(x)
        perda = perda_focal(logits, alvo)
        otim.zero_grad()
        perda.backward()
        otim.step()
        sched.step()
    return modelo.eval()


@torch.no_grad()
def candidatos_bola(modelo: MapaBola, imagens: np.ndarray, device: str = "cuda", max_por_frame: int = 5,
                    lote: int = 4) -> list[np.ndarray]:
    """Picos do mapa de calor por frame: (n, 3) com x, y (escala da entrada) e score, do maior ao menor."""
    saida = []
    for i in range(0, len(imagens), lote):
        p = torch.sigmoid(modelo(preparar(imagens[i:i + lote], device)).float())
        picos = p * (F.max_pool2d(p[:, None], 3, 1, 1)[:, 0] == p)
        for m in picos:
            v, idx = m.flatten().topk(max_por_frame)
            ys, xs = idx // m.shape[1], idx % m.shape[1]
            saida.append(np.stack([(xs.cpu().numpy() + 0.5) * PASSO, (ys.cpu().numpy() + 0.5) * PASSO,
                                   v.cpu().numpy()], axis=1))
    return saida


def carregar_rotulos(saida: Path) -> tuple[pd.DataFrame, np.ndarray, list]:
    """Frames rotulados (scripts/bola_rotular.py): (tabela, imagens RGB na entrada do mapa, centros na
    escala da entrada ou None quando não há bola)."""
    d = pd.read_csv(saida / "amostra.csv").merge(pd.read_csv(saida / "rotulos.csv"), on="arquivo")
    imgs = np.stack([cv2.cvtColor(cv2.resize(cv2.imread(str(saida / "frames" / a)), (LARGURA, ALTURA),
                                             interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB) for a in d["arquivo"]])
    esc = LARGURA / 1920
    return d, imgs, [None if r.x != r.x else (r.x * esc, r.y * esc) for r in d.itertuples()]


def extrair_candidatos(video: Path, modelo: MapaBola, fps: int = 8, minimo: float = 0.1, lote: int = 8,
                       device: str = "cuda") -> pd.DataFrame:
    """Picos de bola de todos os frames do vídeo, amostrados por tempo a `fps`:
    colunas i (frame), x, y (escala da entrada) e score."""
    linhas, fila = [], []

    def processar():
        for (i, _), c in zip(fila, candidatos_bola(modelo, np.stack([f for _, f in fila]), device, lote=len(fila))):
            linhas.extend((i, x, y, s) for x, y, s in c if s >= minimo)
        fila.clear()

    for i, frame in enumerate(iterar_frames(video, fps, LARGURA, ALTURA)):
        fila.append((i, frame[..., ::-1]))
        if len(fila) == lote:
            processar()
    if fila:
        processar()
    return pd.DataFrame(linhas, columns=["i", "x", "y", "score"])


def _estaticos(c: pd.DataFrame, n_frames: int, fps: int, janela_s: float, celula: int) -> np.ndarray:
    """Máscara dos candidatos que ficam no mesmo lugar (célula ou vizinha) em >= metade dos frames de
    +-`janela_s`: bolas reservas na areia, marcas e objetos redondos fixos."""
    cx, cy = (c["x"] // celula).astype(int).to_numpy(), (c["y"] // celula).astype(int).to_numpy()
    celulas = pd.Series(list(zip(cx, cy)))
    k = int(2 * janela_s * fps) | 1
    presenca = {}
    for cel, g in celulas.groupby(celulas):
        if len(g) >= k // 4:  # uma bola parada precisa estar em >= k/2 frames, com folga para falhas
            v = np.zeros(n_frames)
            v[c["i"].to_numpy()[g.index]] = 1
            presenca[cel] = pd.Series(v).rolling(k, center=True, min_periods=1).mean().to_numpy()
    vazio = np.zeros(n_frames)
    return np.array([any(presenca.get((x + dx, y + dy), vazio)[i] >= 0.5 for dx in (-1, 0, 1) for dy in (-1, 0, 1))
                     for x, y, i in zip(cx, cy, c["i"])], bool)


def features_bola(c: pd.DataFrame, n_janelas: int, fps: int = 8, janela_s: float = 0.5, score_min: float = 0.3,
                  estatico_s: float = 10.0, celula: int = 32) -> pd.DataFrame:
    """Features da bola em movimento por janela: nº de bolas por frame, score máximo, fração de frames com
    bola, altura (média e mínima, 0 = topo), velocidade mediana e máxima (largura de quadro por s).
    Candidatos parados (ver `_estaticos`) são ignorados. Janelas sem bola ficam com NaN."""
    por_janela = int(janela_s * fps)
    n_frames = n_janelas * por_janela
    c = c[(c["score"] >= score_min) & (c["i"] < n_frames)].reset_index(drop=True)
    if len(c):
        c = c[~_estaticos(c, n_frames, fps, estatico_s, celula)].reset_index(drop=True)
    melhor = (c.loc[c.groupby("i")["score"].idxmax()] if len(c) else c).set_index("i").astype(float).reindex(range(n_frames))
    n_por_frame = c.groupby("i").size().reindex(range(n_frames), fill_value=0).to_numpy()
    x, y = melhor["x"].to_numpy(), melhor["y"].to_numpy()
    passo = np.hypot(np.diff(x, prepend=np.nan), np.diff(y, prepend=np.nan))
    passo[passo > 0.25 * LARGURA] = np.nan  # salto grande: outra bola ou outro objeto, não é o mesmo deslocamento
    vel = passo * fps / LARGURA

    def por(v, f):
        a = v.reshape(n_janelas, por_janela)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)  # janelas sem bola: fatias só com NaN
            return np.where(np.isnan(a).all(axis=1), np.nan, f(a))

    ha = ~np.isnan(x)
    return pd.DataFrame({
        "bola_n": n_por_frame.reshape(n_janelas, por_janela).mean(axis=1),
        "bola_score": por(melhor["score"].to_numpy(), lambda a: np.nanmax(a, axis=1)),
        "bola_frac": ha.reshape(n_janelas, por_janela).mean(axis=1),
        "bola_y": por(y / ALTURA, lambda a: np.nanmean(a, axis=1)),
        "bola_ymin": por(y / ALTURA, lambda a: np.nanmin(a, axis=1)),
        "bola_vel": por(vel, lambda a: np.nanmedian(a, axis=1)),
        "bola_velmax": por(vel, lambda a: np.nanmax(a, axis=1)),
    })
