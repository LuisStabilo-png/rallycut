"""Compara os cortes detectados no vídeo editado com o gabarito do alinhamento.

Gabarito: início de cada trecho do editado (menos o primeiro) em `trechos.csv`.
Detectados: cortes aceitos em um ou mais `cortes.csv`. Um detectado acerta se estiver a
até --tol segundos de um corte do gabarito (cada corte do gabarito casa uma única vez).

Uso:
    uv run python scripts/fase2_avaliar_cortes.py --trechos outputs/pares/quadra_set1/trechos.csv \
        --cortes outputs/fase1_pares/cortes_pessoas.csv outputs/fase1_pares/cortes_pixels.csv
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def avaliar(gabarito: np.ndarray, detectados: np.ndarray, tol: float) -> dict:
    """Casa cada corte do gabarito com o detectado mais próximo (dentro de tol)."""
    livres = list(range(len(detectados)))
    erros = []
    for g in gabarito:
        if not livres:
            break
        k = min(livres, key=lambda i: abs(detectados[i] - g))
        if abs(detectados[k] - g) <= tol:
            erros.append(detectados[k] - g)
            livres.remove(k)
    acertos = len(erros)
    return {"gabarito": len(gabarito), "detectados": len(detectados), "acertos": acertos,
            "perdidos": len(gabarito) - acertos, "falsos": len(detectados) - acertos,
            "recall": acertos / len(gabarito), "precisao": acertos / max(len(detectados), 1),
            "erro_medio_s": float(np.mean(np.abs(erros))) if erros else float("nan")}


def principal():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--trechos", type=Path, required=True)
    ap.add_argument("--cortes", type=Path, nargs="+", required=True)
    ap.add_argument("--tol", type=float, default=0.25)
    args = ap.parse_args()

    gabarito = pd.read_csv(args.trechos)["editado_ini"].to_numpy()[1:]
    for arq in args.cortes:
        c = pd.read_csv(arq)
        r = avaliar(gabarito, c.loc[c["aceito"], "tempo_s"].to_numpy(), args.tol)
        print(f"{arq.name}: {r['acertos']}/{r['gabarito']} cortes achados (recall {r['recall']:.0%}), "
              f"{r['falsos']} falsos (precisão {r['precisao']:.0%}), erro médio {r['erro_medio_s'] * 1000:.0f} ms")


if __name__ == "__main__":
    principal()
