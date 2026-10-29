"""Estatísticas do par bruto/editado: duração dos rallies e do tempo morto removido.

Uso:
    uv run python scripts/fase2_estatisticas.py --pasta outputs/pares/quadra_set1
"""

import argparse
from pathlib import Path

import pandas as pd

PERCENTIS = [0.05, 0.25, 0.5, 0.75, 0.9, 0.95]


def resumo(x: pd.Series, nome: str) -> str:
    q = x.quantile(PERCENTIS)
    return f"{nome:26s} n={len(x):3d}  " + "  ".join(f"p{int(p * 100)}={v:5.1f}" for p, v in q.items())


def principal():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pasta", type=Path, required=True, help="saída de fase2_alinhar.py")
    args = ap.parse_args()

    trechos = pd.read_csv(args.pasta / "trechos.csv")
    removidos = pd.read_csv(args.pasta / "removidos.csv")
    print(resumo(trechos["duracao_s"], "duração do rally (s)"))
    print(resumo(removidos["duracao_s"], "tempo morto removido (s)"))
    rally, morto = trechos["duracao_s"].sum(), removidos["duracao_s"].sum()
    print(f"\nrally {rally / 60:.1f} min ({rally / (rally + morto):.0%}) | tempo morto {morto / 60:.1f} min")


if __name__ == "__main__":
    principal()
