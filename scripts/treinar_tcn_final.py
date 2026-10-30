"""Treina o TCN final (modo por vídeo, 2 saídas, com features de bola) com todos os pares e salva em
outputs/modelos/tcn_bola.pt (o tcn.pt antigo, sem bola, fica como está).

Precisa de outputs/pares/<par>/bola_candidatos.csv (scripts/bola_extrair.py). Num vídeo novo, os
candidatos de bola vêm de `extrair_candidatos` com o detector outputs/bola/mapa_todos.pt.

Uso:
    uv run python scripts/treinar_tcn_final.py [--sementes 3] [--passos 1500]
"""

import argparse
from pathlib import Path

from rallycut.bola import com_bola
from rallycut.config import carregar_config
from rallycut.features import sufixo
from rallycut.pares import listar_pares, preparar_par
from rallycut.temporal import padronizar, salvar_tcn, treinar_tcn
from rallycut.treino import carregar, dados_tcn


def principal():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--altura", type=int, default=None)
    ap.add_argument("--embeddings", type=int, default=32)
    ap.add_argument("--passos", type=int, default=1500)
    ap.add_argument("--sementes", type=int, default=3)
    ap.add_argument("--excluir", nargs="*", default=[], help="pares que ficam de fora (para validar em vídeo novo)")
    ap.add_argument("--saida", type=Path, default=Path("outputs/modelos/tcn_bola.pt"))
    args = ap.parse_args()

    altura = args.altura or carregar_config("quadra")["features"]["altura"]
    pastas = [preparar_par(p, Path("outputs/pares"), altura=altura) for p in listar_pares() if p.name not in args.excluir]
    d, pca = carregar(pastas, args.embeddings, sufixo(altura))
    pares, colunas, Xs, ys = dados_tcn(com_bola(d))
    Z = padronizar(Xs, "video", list(range(len(pares))))
    modelos = [treinar_tcn(Z, ys, args.passos, semente=s) for s in range(args.sementes)]
    args.saida.parent.mkdir(parents=True, exist_ok=True)
    salvar_tcn(modelos, colunas, pca, args.saida)
    print(f"{len(pares)} pares, {len(colunas)} features, {len(modelos)} sementes -> {args.saida}")


if __name__ == "__main__":
    principal()
