"""Alinha o vídeo editado ao bruto e grava os trechos (rallies) e o tempo morto removido.

Saída em --saida: trechos.csv/json, removidos.csv e amostra.jpg (frame do editado ao
lado do frame do bruto no mesmo ponto: devem ser idênticos).

Uso:
    uv run python scripts/fase2_alinhar.py --bruto <bruto.mp4> --editado <editado.mp4> \
        --saida outputs/pares/quadra_set1 --modalidade quadra
"""

import argparse
from pathlib import Path

from rallycut.alinhamento import alinhar_par
from rallycut.config import MODALIDADES, carregar_config


def principal():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bruto", type=Path, required=True)
    ap.add_argument("--editado", type=Path, required=True)
    ap.add_argument("--saida", type=Path, required=True)
    ap.add_argument("--modalidade", choices=MODALIDADES, default="quadra")
    args = ap.parse_args()

    t = alinhar_par(args.bruto, args.editado, args.saida, carregar_config(args.modalidade))
    d = t["duracao_s"]
    print(f"{len(t)} trechos | mantido {d.sum() / 60:.1f} min | duração mediana {d.median():.1f}s "
          f"(min {d.min():.1f}, máx {d.max():.1f}) | custo mediano {t['custo'].median():.2f}, máx {t['custo'].max():.2f} "
          f"| ajuste de áudio máx {t['ajuste_s'].abs().max():.2f}s")


if __name__ == "__main__":
    principal()
