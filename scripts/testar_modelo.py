"""Testa um TCN salvo (ex.: outputs/modelos/tcn_bola.pt) em pares com gabarito, com o limiar e as margens
da config: é o que o `cortar.py` faria nesses vídeos. Serve para medir um modelo em jogos que ele não viu.

Usa as features e a bola em cache de outputs/pares/<par>/ (preparar_par + bola_extrair).

Uso:
    uv run python scripts/testar_modelo.py outputs/modelos/tcn_bola.pt praia_set6 praia_set7 [--csv saida.csv]
"""

import argparse
from pathlib import Path

import pandas as pd

from rallycut.config import carregar_config
from rallycut.deteccao import avaliar, segmentar
from rallycut.features import JANELA_S
from rallycut.inferencia import tabela_de_features
from rallycut.pares import bruto_do_par
from rallycut.temporal import carregar_tcn, prever_video


def principal():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("modelo", type=Path)
    ap.add_argument("pares", nargs="+")
    ap.add_argument("--csv", type=Path, default=None)
    args = ap.parse_args()

    modelos, colunas, pca = carregar_tcn(args.modelo)
    linhas = []
    for par in args.pares:
        mod, saida = par.split("_")[0], Path("outputs/pares") / par
        X = tabela_de_features(bruto_do_par(Path("data/pares") / par), saida, mod, Path("outputs/bola/mapa_todos.pt"),
                               pca, colunas)
        p = prever_video(modelos, X)
        trechos = pd.read_csv(saida / "trechos.csv")
        r = avaliar(trechos, segmentar(p, carregar_config(mod)["deteccao"], len(p) * JANELA_S))
        razao = r["mantido_min"] / (trechos["duracao_s"].sum() / 60)
        linhas.append({"par": par, **r, "razao": razao})
        print(f"{par}: início preservado {r['inicios_preservados']:.0%} | perdidos {r['perdidos']}/{r['rallies']} | "
              f"falsos {r['falsos']} | saída {razao:.2f}x o gabarito", flush=True)
    if args.csv:
        pd.DataFrame(linhas).to_csv(args.csv, index=False)


if __name__ == "__main__":
    principal()
