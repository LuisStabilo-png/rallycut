"""Extrai os candidatos de bola de cada par (Portão 2): outputs/pares/<par>/bola_candidatos.csv.

Detector sem vazamento: cada jogo de praia usa um detector treinado SEM os frames rotulados dele
(outputs/bola/mapa_<jogo>.pt); a quadra usa um treinado com todos os jogos (mapa_todos.pt).
Os detectores e os candidatos ficam em cache.

Uso:
    uv run python scripts/bola_extrair.py [--fps 8] [--passos 1200]
"""

import argparse
import time
from pathlib import Path

import numpy as np
import torch

from rallycut.bola import MapaBola, carregar_mapa, carregar_rotulos, extrair_candidatos, treinar_mapa
from rallycut.pares import bruto_do_par, listar_pares

SAIDA = Path("outputs/bola")


def detector(nome: str, d, imgs, centros, passos: int) -> MapaBola:
    """Carrega (ou treina e guarda) o detector que não viu o jogo `nome` (`todos` = viu todos)."""
    caminho = SAIDA / f"mapa_{nome}.pt"
    if caminho.exists():
        return carregar_mapa(caminho)
    t0 = time.time()
    treino = np.flatnonzero((d["par"] != nome).to_numpy())
    modelo = treinar_mapa(imgs[treino], [centros[i] for i in treino], passos)
    torch.save(modelo.state_dict(), caminho)
    print(f"  detector {nome}: treinado em {time.time() - t0:.0f} s", flush=True)
    return modelo


def principal():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fps", type=int, default=8)
    ap.add_argument("--passos", type=int, default=1200)
    args = ap.parse_args()

    d, imgs, centros = carregar_rotulos(SAIDA)
    for pasta in listar_pares():
        saida = Path("outputs/pares") / pasta.name / "bola_candidatos.csv"
        if saida.exists():
            continue
        modelo = detector(pasta.name if pasta.name in set(d["par"]) else "todos", d, imgs, centros, args.passos)
        t0 = time.time()
        c = extrair_candidatos(bruto_do_par(pasta), modelo, args.fps)
        c.round(2).to_csv(saida, index=False)
        print(f"{pasta.name}: {len(c)} candidatos, {(time.time() - t0) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    principal()
