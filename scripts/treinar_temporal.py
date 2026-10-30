"""Compara o modelo temporal (TCN) com o LightGBM, deixando um jogo de fora de cada vez.

Os modelos calibram a probabilidade de formas diferentes, então a comparação usa o mesmo
tamanho de saída: o início preservado quando o vídeo final tem `--razao` x o tempo do gabarito.

Uso:
    uv run python scripts/treinar_temporal.py [--passos 1500] [--modos global video] [--sementes 3] [--histerese 1 0.5]

O "ensemble" é a média dos percentis (por jogo) do LightGBM e dos TCNs, semente a semente.
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
    ap.add_argument("--sementes", type=int, default=1, help="repetições do treino do TCN")
    ap.add_argument("--histerese", type=float, nargs="+", default=[1.0, 0.5],
                    help="fatores de histerese da segmentação (1 = sem)")
    args = ap.parse_args()

    altura = args.altura or carregar_config("quadra")["features"]["altura"]
    pastas = [preparar_par(p, Path("outputs/pares"), altura=altura) for p in listar_pares()]
    d, _ = carregar(pastas, args.embeddings, sufixo(altura))
    gbm_cols = [c for c in d.columns if c not in ("t", "rotulo", "par", "bloco")]
    brutas = [c for c in gbm_cols if not c.endswith(("_mean10", "_mean30", "_std10", "_std30"))]
    pares = list(d["par"].unique())
    Xs = [d.loc[d["par"] == p, brutas].to_numpy(float) for p in pares]
    ys = [d.loc[d["par"] == p, "rotulo"].to_numpy(float) for p in pares]

    # modelos[nome][par] = previsões, uma por semente (o LightGBM é determinístico: uma só)
    modelos = {"lightgbm": []}
    for p in pares:
        fora = d["par"] == p
        m = treinar_gbm(d.loc[~fora, gbm_cols], d.loc[~fora, "rotulo"])
        modelos["lightgbm"].append([m.predict(d.loc[fora, gbm_cols])])
    for modo in args.modos:
        modelos[f"tcn-{modo}"] = []
        for i in range(len(pares)):
            treino = [j for j in range(len(pares)) if j != i]
            Z = padronizar(Xs, modo, treino)
            modelos[f"tcn-{modo}"].append([
                prever(treinar_tcn([Z[j] for j in treino], [ys[j] for j in treino], args.passos, semente=s), Z[i])
                for s in range(args.sementes)])

    modelos["ensemble"] = [[np.mean([pd.Series(m[i][s if len(m[i]) > 1 else 0]).rank(pct=True) for m in modelos.values()],
                                    axis=0) for s in range(args.sementes)] for i in range(len(pares))]

    for h in args.histerese:
        print(f"\njogo de fora; histerese {h}; início preservado com vídeo final = {args.razao}x o gabarito "
              f"(margem pré 4 s); {args.sementes} semente(s): média ± desvio")
        print(f"{'par':12s}" + "".join(f"| {n:^28s}" for n in modelos))
        por_modalidade = {n: {"quadra": [], "praia": []} for n in modelos}
        for i, par in enumerate(pares):
            trechos, mod = pd.read_csv(f"outputs/pares/{par}/trechos.csv"), par.split("_")[0]
            celulas = []
            for n, ps in modelos.items():
                inicios = np.array([no_orcamento(trechos, p, mod, args.razao, h)[1]["inicios_preservados"] for p in ps[i]])
                por_modalidade[n][mod].append(inicios)
                celulas.append(f"AUC {np.mean([auc(p, ys[i]) for p in ps[i]]):.2f}  início {inicios.mean():4.0%} ±{inicios.std():.0%}")
            print(f"{par:12s}" + "".join(f"| {c:^28s}" for c in celulas))
        for mod in ("quadra", "praia"):
            por_semente = [np.mean(v[mod], axis=0) for v in por_modalidade.values()]
            print(f"{'média ' + mod:12s}" + "".join(f"| {m.mean():^20.0%} ±{m.std():<6.0%}" for m in por_semente))


if __name__ == "__main__":
    principal()
