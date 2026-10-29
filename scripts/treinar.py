"""Treina e avalia o detector de rallies com todos os pares bruto/editado em data/pares/.

Tudo que falta é feito sozinho (alinhamento e features, que ficam em cache em outputs/pares/).
Cada par novo colocado em data/pares/<quadra|praia>_<nome>/ entra no próximo treino.

Avaliação (mede o que importa: % de inícios de rally preservados):
  blocos   - validação cruzada por blocos de 10 min do mesmo jogo (mesma modalidade/ginásio);
  jogo     - treina sem um jogo e testa nele (generalização para um jogo novo).
Saída em outputs/modelos/: modelo.txt (LightGBM) e metricas.json.

Uso:
    uv run python scripts/treinar.py
"""

import argparse
import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from rallycut.config import carregar_config
from rallycut.deteccao import BASE, avaliar, com_contexto, segmentar
from rallycut.features import JANELA_S
from rallycut.pares import listar_pares, preparar_par

BLOCO_S = 600


def carregar(pastas: list[Path], k_emb: int) -> tuple[pd.DataFrame, dict]:
    """Junta as janelas de todos os pares, com as `k_emb` componentes principais dos embeddings
    visuais (PCA ajustada em todos os pares) e o contexto temporal de cada feature."""
    dfs = [pd.read_csv(p / "janelas.csv") for p in pastas]
    pca, cols = {}, []
    if k_emb:
        embs = [np.load(p / "embeddings.npy").astype(np.float32) for p in pastas]
        E = np.vstack(embs)
        pca = {"media": E.mean(axis=0), "componentes": np.linalg.svd(E - E.mean(axis=0), full_matrices=False)[2][:k_emb]}
        cols = [f"emb{i}" for i in range(k_emb)]
        for df, e in zip(dfs, embs):
            df[cols] = (e - pca["media"]) @ pca["componentes"].T
    for df, p in zip(dfs, pastas):
        df["par"], df["eh_praia"] = p.name, int(p.name.startswith("praia"))
        df["bloco"] = (df["t"] // BLOCO_S).astype(int)
    return pd.concat([com_contexto(df, colunas=BASE + cols) for df in dfs], ignore_index=True), pca


PARAMS = {"objective": "binary", "learning_rate": 0.05, "num_leaves": 15, "min_data_in_leaf": 20,
          "bagging_fraction": 0.8, "bagging_freq": 1, "feature_fraction": 0.8, "verbose": -1}


def treinar(X: pd.DataFrame, y: pd.Series) -> lgb.Booster:
    return lgb.train(PARAMS, lgb.Dataset(X, y), num_boost_round=300)


def relatorio(esquema: str, d: pd.DataFrame, proba: np.ndarray, saida_base: Path) -> list[dict]:
    """Segmenta e avalia cada par a partir das probabilidades de `d`."""
    linhas = []
    for par, g in d.assign(p=proba).groupby("par"):
        cfg = carregar_config(par.split("_")[0])["deteccao"]
        trechos = pd.read_csv(saida_base / par / "trechos.csv")
        r = avaliar(trechos, segmentar(g["p"].to_numpy(), cfg, len(g) * JANELA_S))
        linhas.append({"esquema": esquema, "par": par, **r})
        print(f"{esquema:7s} {par:14s} início preservado {r['inicios_preservados']:4.0%} | "
              f"perdidos {r['perdidos']}/{r['rallies']} | falsos {r['falsos']} | "
              f"tempo morto sobrando {r['morto_residual']:3.0%} | mantido {r['mantido_min']:.1f} min "
              f"(gabarito {trechos['duracao_s'].sum() / 60:.1f})")
    return linhas


def principal():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pares", type=Path, default=Path("data/pares"))
    ap.add_argument("--saida", type=Path, default=Path("outputs/modelos"))
    ap.add_argument("--embeddings", type=int, default=32, help="componentes dos embeddings visuais (0 = não usa)")
    args = ap.parse_args()

    saida_base = Path("outputs/pares")
    pastas = [preparar_par(p, saida_base) for p in listar_pares(args.pares)]
    d, pca = carregar(pastas, args.embeddings)
    colunas = [c for c in d.columns if c not in ("t", "rotulo", "par", "bloco")]
    print(f"{len(pastas)} pares, {len(d)} janelas ({d['rotulo'].mean():.0%} rally), {len(colunas)} features\n")

    resultados = []
    # blocos de 10 min: cada bloco fica fora uma vez (até 5 grupos)
    blocos = d.groupby(["par", "bloco"]).ngroup()
    grupo = blocos % min(5, blocos.nunique())
    oof = np.zeros(len(d))
    for g in np.unique(grupo):
        m = treinar(d.loc[grupo != g, colunas], d.loc[grupo != g, "rotulo"])
        oof[grupo == g] = m.predict(d.loc[grupo == g, colunas])
    resultados += relatorio("blocos", d, oof, saida_base)

    if len(pastas) > 1:
        oof = np.zeros(len(d))
        for par in d["par"].unique():
            fora = d["par"] == par
            m = treinar(d.loc[~fora, colunas], d.loc[~fora, "rotulo"])
            oof[fora] = m.predict(d.loc[fora, colunas])
        resultados += relatorio("jogo", d, oof, saida_base)

    final = treinar(d[colunas], d["rotulo"])
    args.saida.mkdir(parents=True, exist_ok=True)
    final.save_model(str(args.saida / "modelo.txt"))
    if pca:
        np.savez(args.saida / "pca.npz", **pca)
    imp = pd.Series(final.feature_importance("gain"), colunas).sort_values(ascending=False)
    (args.saida / "metricas.json").write_text(json.dumps(
        {"resultados": resultados, "importancia": imp.head(10).to_dict(), "colunas": colunas}, indent=1))
    print("\nfeatures mais usadas:", ", ".join(f"{k}" for k in imp.head(6).index))
    print(f"modelo e métricas em {args.saida}")


if __name__ == "__main__":
    principal()
