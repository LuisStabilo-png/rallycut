"""Sorteia frames dos brutos de praia para medir a detecção da bola (Portão 1 do experimento de bola).

Por par: `--n` frames, metade dentro de rallies e metade no tempo morto (gabarito do alinhamento),
extraídos por TEMPO a 1920x1080. Saída em outputs/bola/: frames/*.jpg e amostra.csv.

Uso:
    uv run python scripts/bola_amostrar.py [--n 60] [--modalidade praia]
"""

import argparse
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

from rallycut.pares import listar_pares

SAIDA = Path("outputs/bola")


def sortear(intervalos: pd.DataFrame, n: int, rng, margem_s: float = 1.0) -> np.ndarray:
    """`n` instantes uniformes (sem repetir, a >= 1 s um do outro) dentro dos intervalos `bruto_ini`/`bruto_fim`."""
    ini, fim = intervalos["bruto_ini"].to_numpy() + margem_s, intervalos["bruto_fim"].to_numpy() - margem_s
    ok = fim > ini
    ini, fim = ini[ok], fim[ok]
    tempos: list[float] = []
    while len(tempos) < n:
        i = rng.choice(len(ini), p=(fim - ini) / (fim - ini).sum())
        t = rng.uniform(ini[i], fim[i])
        if all(abs(t - u) >= 1.0 for u in tempos):
            tempos.append(t)
    return np.sort(tempos)


def extrair(video: Path, t: float, destino: Path):
    subprocess.run(["ffmpeg", "-nostdin", "-loglevel", "error", "-y", "-ss", f"{t:.3f}", "-i", str(video),
                    "-frames:v", "1", "-vf", "scale=1920:1080", "-q:v", "2", str(destino)], check=True)


def principal():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=60, help="frames por par (metade rally, metade tempo morto)")
    ap.add_argument("--modalidade", default="praia")
    ap.add_argument("--semente", type=int, default=0)
    args = ap.parse_args()

    rng = np.random.default_rng(args.semente)
    (SAIDA / "frames").mkdir(parents=True, exist_ok=True)
    linhas = []
    for pasta in listar_pares():
        if not pasta.name.startswith(args.modalidade):
            continue
        bruto = next(pasta.glob("*bruto*"))
        gab = Path("outputs/pares") / pasta.name
        for classe, csv in (("rally", "trechos.csv"), ("morto", "removidos.csv")):
            for t in sortear(pd.read_csv(gab / csv), args.n // 2, rng):
                arq = f"{pasta.name}_{t:08.2f}_{classe}.jpg"
                extrair(bruto, t, SAIDA / "frames" / arq)
                linhas.append({"par": pasta.name, "t": round(t, 3), "classe": classe, "arquivo": arq})
        print(f"{pasta.name}: ok")
    pd.DataFrame(linhas).to_csv(SAIDA / "amostra.csv", index=False)
    print(f"{len(linhas)} frames em {SAIDA / 'frames'}")


if __name__ == "__main__":
    principal()
