"""Fase 1: testa se a continuidade dos jogadores separa cortes reais de falsos.

Usa os cortes conferidos manualmente em `outputs/fase1/validacao/rotulos.csv`
(colunas video, tempo_s, frame, real) e grava `rotulos_pessoas.csv` com as métricas.
Imprime a AUC de cada métrica (1.0 = separa perfeitamente; 0.5 = não ajuda).

Uso:
    uv run python scripts/fase1_testar_pessoas.py
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from rallycut.pessoas import carregar_detector, verificar_cortes

METRICAS = ["perto_max", "perto_med", "maior", "frac_estavel", "custo_hung"]


def auc(valores: pd.Series, real: pd.Series) -> float:
    """P(valor de um corte real > valor de um falso), com empates valendo 0,5."""
    ok = valores.notna()
    pos, neg = valores[ok & real].to_numpy(), valores[ok & ~real].to_numpy()
    if not len(pos) or not len(neg):
        return float("nan")
    return float(((pos[:, None] > neg[None, :]).mean() + 0.5 * (pos[:, None] == neg[None, :]).mean()))


def principal():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rotulos", type=Path, default=Path("outputs/fase1/validacao/rotulos.csv"))
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    d = pd.read_csv(args.rotulos)
    modelo = carregar_detector(args.device)
    d = d.drop(columns=[c for c in d.columns if c in METRICAS + ["n_antes", "n_depois"]])
    partes = []
    for video, g in d.groupby("video"):
        caminho = Path(f"data/editados/{video.split('_')[0]}/{video}.mp4")
        partes.append(verificar_cortes(caminho, g["frame"].tolist(), modelo, args.device, progresso=None))
    d = d.merge(pd.concat(partes), on="frame", how="left")
    d.to_csv(args.rotulos.with_name("rotulos_pessoas.csv"), index=False)

    print(f"{len(d)} cortes: {int(d.real.sum())} reais, {int((~d.real).sum())} falsos; "
          f"sem detecção: {int(d.perto_max.isna().sum())}")
    print("\nAUC (reais > falsos):")
    for nome in METRICAS + ["dm", "fr", "zv", "za"]:
        print(f"  {nome:13s} {auc(d[nome], d.real):.3f}")
    print("\nMediana por classe:")
    print(d.groupby("real")[METRICAS + ["n_antes", "n_depois"]].median().round(3).to_string())


if __name__ == "__main__":
    principal()
