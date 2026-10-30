"""Dados dos pares e métricas comuns aos scripts de treino."""

from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from rallycut.config import carregar_config
from rallycut.deteccao import BASE, avaliar, com_contexto, segmentar
from rallycut.features import JANELA_S

BLOCO_S = 600
PARAMS = {"objective": "binary", "learning_rate": 0.05, "num_leaves": 15, "min_data_in_leaf": 20,
          "bagging_fraction": 0.8, "bagging_freq": 1, "feature_fraction": 0.8, "verbose": -1}


def carregar(pastas: list[Path], k_emb: int, suf: str = "") -> tuple[pd.DataFrame, dict]:
    """Junta as janelas de todos os pares, com as `k_emb` componentes principais dos embeddings
    visuais (PCA ajustada em todos os pares) e o contexto temporal de cada feature."""
    dfs = [pd.read_csv(p / f"janelas{suf}.csv") for p in pastas]
    pca, cols = {}, []
    if k_emb:
        embs = [np.load(p / f"embeddings{suf}.npy").astype(np.float32) for p in pastas]
        E = np.vstack(embs)
        pca = {"media": E.mean(axis=0), "componentes": np.linalg.svd(E - E.mean(axis=0), full_matrices=False)[2][:k_emb]}
        cols = [f"emb{i}" for i in range(k_emb)]
        for df, e in zip(dfs, embs):
            df[cols] = (e - pca["media"]) @ pca["componentes"].T
    for df, p in zip(dfs, pastas):
        df["par"], df["eh_praia"] = p.name, int(p.name.startswith("praia"))
        df["bloco"] = (df["t"] // BLOCO_S).astype(int)
    return pd.concat([com_contexto(df, colunas=BASE + cols) for df in dfs], ignore_index=True), pca


def treinar_gbm(X: pd.DataFrame, y: pd.Series) -> lgb.Booster:
    return lgb.train(PARAMS, lgb.Dataset(X, y), num_boost_round=300)


def auc(p: np.ndarray, y: np.ndarray) -> float:
    """P(janela de rally ter probabilidade maior que a de tempo morto)."""
    r = pd.Series(p).rank().to_numpy()
    n1 = y.sum()
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * (len(y) - n1)))


def no_orcamento(trechos: pd.DataFrame, p: np.ndarray, modalidade: str, razao: float = 1.5,
                 histerese: float | None = None):
    """Limiar para o vídeo final ficar com `razao` x o tempo do gabarito (margem pré de 4 s).

    Compara modelos de calibração diferente pelo mesmo tamanho de saída. Devolve (limiar, métricas)."""
    cfg = {**carregar_config(modalidade)["deteccao"], "margem_pre_s": 4.0}
    if histerese is not None:
        cfg["histerese"] = histerese
    alvo = trechos["duracao_s"].sum() / 60 * razao
    candidatos = [(abs(r["mantido_min"] - alvo), lim, r)
                  for lim in np.arange(0.02, 0.8, 0.02)
                  for r in [avaliar(trechos, segmentar(p, {**cfg, "limiar": lim}, len(p) * JANELA_S))]]
    _, lim, r = min(candidatos, key=lambda c: c[0])
    return float(lim), r
