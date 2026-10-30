"""Modelo temporal: rede convolucional 1D (TCN, não causal) sobre a sequência de janelas de 0,5 s.

Com dilatações 1, 2, 4, 8 e 16 e kernel 5, cada janela enxerga ~±1 min ao redor, o que permite
reconhecer o ritmo do jogo (saque, troca de bolas, parada) em vez de olhar uma janela isolada.
"""

from pathlib import Path

import numpy as np
import torch
from torch import nn


class Bloco(nn.Module):
    def __init__(self, canais: int, dilatacao: int, p: float):
        super().__init__()
        self.f = nn.Sequential(nn.Conv1d(canais, canais, 5, padding=2 * dilatacao, dilation=dilatacao),
                               nn.GELU(), nn.Dropout(p), nn.Conv1d(canais, canais, 1))
        self.norma = nn.GroupNorm(1, canais)

    def forward(self, x):
        return self.norma(x + self.f(x))


class TCN(nn.Module):
    def __init__(self, n_entradas: int, n_saidas: int = 2, canais: int = 64, dilatacoes=(1, 2, 4, 8, 16),
                 p: float = 0.3):
        super().__init__()
        self.entrada = nn.Conv1d(n_entradas, canais, 1)
        self.blocos = nn.Sequential(*[Bloco(canais, d, p) for d in dilatacoes])
        self.saida = nn.Conv1d(canais, n_saidas, 1)

    def forward(self, x):  # (lote, canais, tempo) -> logits (lote, saídas, tempo)
        return self.saida(self.blocos(self.entrada(x)))


def padronizar(Xs: list[np.ndarray], modo: str, treino: list[int]) -> list[np.ndarray]:
    """Padroniza as features. `global`: média e desvio dos jogos de treino; `video`: de cada jogo
    (tira o "sotaque" do local). NaN vira 0 depois de padronizar."""
    if modo == "global":
        ref = np.vstack([Xs[i] for i in treino])
        mu, sd = np.nanmean(ref, axis=0), np.nanstd(ref, axis=0) + 1e-6
        saida = [(x - mu) / sd for x in Xs]
    else:
        saida = [(x - np.nanmedian(x, axis=0)) / (np.nanstd(x, axis=0) + 1e-6) for x in Xs]
    return [np.nan_to_num(x).astype(np.float32) for x in saida]


def treinar_tcn(Xs: list[np.ndarray], ys: list[np.ndarray], passos: int = 1500, comp: int = 240,
                lote: int = 16, device: str = "cuda", semente: int = 0, peso_inicio: float = 5.0) -> TCN:
    """Treina em trechos aleatórios (2 min) dos jogos, com aumento de dados que imita a troca de
    local: ganho e deslocamento aleatórios por canal, ruído e canais zerados.

    Cada `ys[i]` é (janelas, 2): coluna 0 = rally, coluna 1 = entorno do início (raro, por isso
    `peso_inicio` nos positivos)."""
    torch.manual_seed(semente)
    rng = np.random.default_rng(semente)
    modelo = TCN(Xs[0].shape[1], ys[0].shape[1]).to(device)
    pos_weight = torch.tensor([1.0, peso_inicio], device=device).view(-1, 1)
    otimizador = torch.optim.AdamW(modelo.parameters(), lr=1e-3, weight_decay=1e-2)
    pesos = np.array([len(x) for x in Xs], float)
    pesos /= pesos.sum()
    for _ in range(passos):
        xb, yb = [], []
        for _ in range(lote):
            i = rng.choice(len(Xs), p=pesos)
            a = rng.integers(0, len(Xs[i]) - comp)
            c = Xs[i].shape[1]
            x = Xs[i][a:a + comp] * rng.normal(1, 0.2, c) + rng.normal(0, 0.2, c) + rng.normal(0, 0.1, (comp, c))
            xb.append(x * (rng.random(c) > 0.1))
            yb.append(ys[i][a:a + comp])
        x = torch.tensor(np.stack(xb), dtype=torch.float32, device=device).transpose(1, 2)
        y = torch.tensor(np.stack(yb), dtype=torch.float32, device=device).transpose(1, 2)
        perda = nn.functional.binary_cross_entropy_with_logits(modelo(x), y, pos_weight=pos_weight)
        otimizador.zero_grad()
        perda.backward()
        otimizador.step()
    return modelo.eval()


@torch.no_grad()
def prever(modelo: TCN, X: np.ndarray, device: str = "cuda") -> np.ndarray:
    """Probabilidades (janelas, saídas)."""
    x = torch.tensor(X, dtype=torch.float32, device=device).T[None]
    return torch.sigmoid(modelo(x))[0].T.cpu().numpy()


def salvar_tcn(modelos: list[TCN], colunas: list[str], pca: dict, caminho: Path):
    """Guarda as sementes, a ordem das colunas e a PCA dos embeddings (`media`, `componentes`), que um
    vídeo novo precisa para ser projetado como no treino."""
    torch.save({"colunas": colunas, "n_saidas": modelos[0].saida.out_channels,
                "estados": [m.state_dict() for m in modelos],
                "pca": {k: torch.from_numpy(np.asarray(v)) for k, v in pca.items()}}, caminho)


def carregar_tcn(caminho: Path, device: str = "cuda") -> tuple[list[TCN], list[str], dict]:
    d = torch.load(caminho, map_location=device)
    modelos = []
    for estado in d["estados"]:
        m = TCN(len(d["colunas"]), d["n_saidas"]).to(device)
        m.load_state_dict(estado)
        modelos.append(m.eval())
    return modelos, d["colunas"], {k: v.cpu().numpy() for k, v in d["pca"].items()}


def prever_video(modelos: list[TCN], X: np.ndarray, device: str = "cuda") -> np.ndarray:
    """Probabilidade de rally por janela de um vídeo (features brutas, na ordem de `colunas`):
    média das sementes do máximo entre as saídas de rally e de início."""
    Z = padronizar([X], "video", [])[0]
    return np.mean([prever(m, Z, device).max(axis=1) for m in modelos], axis=0)
