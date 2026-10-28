"""Fase 1: gera o relatório HTML de validação de cortes de um vídeo editado.

Lê `outputs/fase1/<nome>/cortes.csv` e `trechos.csv` (gerados por fase1_detectar_cortes.py)
e grava `relatorio.html` na mesma pasta. Se existir `correcoes.csv`, aplica as correções
e regrava cortes.csv/trechos.csv antes de gerar o relatório (use --sem-correcoes para ignorar).

Uso:
    uv run python scripts/fase1_relatorio.py data/editados/quadra/<video>.mp4
"""

import argparse
from pathlib import Path

import pandas as pd

from rallycut.cortes import aplicar_correcoes, trechos_entre_cortes
from rallycut.media import sondar_video
from rallycut.relatorio import gerar_html


def principal():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("video", type=Path)
    ap.add_argument("--saida", type=Path, default=Path("outputs/fase1"))
    ap.add_argument("--amostra-alta", type=int, default=30)
    ap.add_argument("--sem-correcoes", action="store_true")
    args = ap.parse_args()

    pasta = args.saida / args.video.stem
    cortes = pd.read_csv(pasta / "cortes.csv")
    info = sondar_video(args.video)

    arq_corr = pasta / "correcoes.csv"
    if arq_corr.exists() and not args.sem_correcoes:
        cortes = aplicar_correcoes(cortes, pd.read_csv(arq_corr), info["fps"])
        cortes.to_csv(pasta / "cortes_corrigidos.csv", index=False)
        print(f"correções aplicadas: {len(pd.read_csv(arq_corr))}")
    trechos = trechos_entre_cortes(cortes.loc[cortes["aceito"], "tempo_s"].to_numpy(), info["duracao_s"])
    trechos.to_csv(pasta / "trechos.csv", index=False)

    destino = pasta / "relatorio.html"
    gerar_html(args.video, cortes, trechos, info["fps"], destino, n_amostra_alta=args.amostra_alta)
    print(f"relatório: {destino.resolve()}")


if __name__ == "__main__":
    principal()
