"""Compara o modelo temporal (TCN) com o LightGBM, deixando um jogo de fora de cada vez.

Os modelos calibram a probabilidade de formas diferentes, então a comparação usa o mesmo
tamanho de saída: o início preservado quando o vídeo final tem `--razao` x o tempo do gabarito.

Uso:
    uv run python scripts/treinar_temporal.py [--passos 1500] [--modos global video]
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from rallycut.config import carregar_config
from rallycut.features import sufixo
from rallycut.pares import listar_pares, preparar_par
from rallycut.temporal import padronizar, prever, treinar_tcn
from rallycut.treino import auc, carregar, no_orcamento, treinar_gbm


def principal():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--altura", type=int, default=None)
    ap.add_argument("--embeddings", type=int, default=32)
    ap.add_argument("--passos", type=int, default=1500)
    ap.add_argument("--modos", nargs="+", default=["global", "video"], choices=["global", "video"])
    ap.add_argument("--razao", type=float, default=1.5)
    args = ap.parse_args()

    altura = args.altura or carregar_config("quadra")["features"]["altura"]
    pastas = [preparar_par(p, Path("outputs/pares"), altura=altura) for p in listar_pares()]
    d, _ = carregar(pastas, args.embeddings, sufixo(altura))
    gbm_cols = [c for c in d.columns if c not in ("t", "rotulo", "par", "bloco")]
    brutas = [c for c in gbm_cols if not c.endswith(("_mean10", "_mean30", "_std10", "_std30"))]
    pares = list(d["par"].unique())
    Xs = [d.loc[d["par"] == p, brutas].to_numpy(float) for p in pares]
    ys = [d.loc[d["par"] == p, "rotulo"].to_numpy(float) for p in pares]

    modelos = {"lightgbm": []}
    for p in pares:
        fora = d["par"] == p
        m = treinar_gbm(d.loc[~fora, gbm_cols], d.loc[~fora, "rotulo"])
        modelos["lightgbm"].append(m.predict(d.loc[fora, gbm_cols]))
    for modo in args.modos:
        modelos[f"tcn-{modo}"] = []
        for i in range(len(pares)):
            treino = [j for j in range(len(pares)) if j != i]
            Z = padronizar(Xs, modo, treino)
            m = treinar_tcn([Z[j] for j in treino], [ys[j] for j in treino], args.passos)
            modelos[f"tcn-{modo}"].append(prever(m, Z[i]))

    print(f"\njogo de fora; início preservado com vídeo final = {args.razao}x o gabarito (margem pré 4 s)")
    print(f"{'par':12s}" + "".join(f"| {n:^24s}" for n in modelos))
    por_modalidade = {n: {"quadra": [], "praia": []} for n in modelos}
    for i, par in enumerate(pares):
        trechos, mod = pd.read_csv(f"outputs/pares/{par}/trechos.csv"), par.split("_")[0]
        celulas = []
        for n, ps in modelos.items():
            _, r = no_orcamento(trechos, ps[i], mod, args.razao)
            por_modalidade[n][mod].append(r["inicios_preservados"])
            celulas.append(f"AUC {auc(ps[i], ys[i]):.2f}  início {r['inicios_preservados']:4.0%}")
        print(f"{par:12s}" + "".join(f"| {c:^24s}" for c in celulas))
    print(f"{'média quadra':12s}" + "".join(f"| {np.mean(v['quadra']):^24.0%}" for v in por_modalidade.values()))
    print(f"{'média praia':12s}" + "".join(f"| {np.mean(v['praia']):^24.0%}" for v in por_modalidade.values()))


if __name__ == "__main__":
    principal()
