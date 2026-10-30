"""Portão 2: as features de bola separam rally de tempo morto? (jogo de fora, LightGBM)

AUC por par com três conjuntos de features: só bola, só as atuais (movimento, jogadores, áudio,
embeddings) e atuais + bola. Critério do Portão 2: AUC médio das features de bola sozinhas >= 0,65 na praia.

Uso:
    uv run python scripts/bola_sinal.py [--altura 720]
"""

import argparse
from pathlib import Path

import pandas as pd

from rallycut.bola import com_bola
from rallycut.config import carregar_config
from rallycut.features import sufixo
from rallycut.pares import listar_pares, preparar_par
from rallycut.treino import auc, carregar, treinar_gbm


def principal():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--altura", type=int, default=None)
    ap.add_argument("--embeddings", type=int, default=32)
    args = ap.parse_args()

    altura = args.altura or carregar_config("quadra")["features"]["altura"]
    pastas = [preparar_par(p, Path("outputs/pares"), altura=altura) for p in listar_pares()]
    d, _ = carregar(pastas, args.embeddings, sufixo(altura))
    d = com_bola(d)
    todas = [c for c in d.columns if c not in ("t", "rotulo", "par", "bloco")]
    so_bola = [c for c in todas if c.startswith("bola_")]
    atuais = [c for c in todas if c not in so_bola]
    conjuntos = {"só bola": so_bola, "atuais": atuais, "atuais+bola": todas}

    linhas = []
    for par in d["par"].unique():
        fora = d["par"] == par
        L = {"par": par}
        for nome, cols in conjuntos.items():
            m = treinar_gbm(d.loc[~fora, cols], d.loc[~fora, "rotulo"])
            L[nome] = auc(m.predict(d.loc[fora, cols]), d.loc[fora, "rotulo"].to_numpy())
        linhas.append(L)
    r = pd.DataFrame(linhas).set_index("par")
    print("AUC, jogo de fora (LightGBM)\n", r.round(2).to_string())
    for mod in ("praia", "quadra"):
        print(f"média {mod}: " + " | ".join(f"{n} {r.loc[r.index.str.startswith(mod), n].mean():.2f}" for n in r.columns))
    r.to_csv("outputs/bola/sinal.csv")


if __name__ == "__main__":
    principal()
