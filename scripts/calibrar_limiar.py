"""Calibra o limiar de corte para vídeos sem gabarito, a partir das previsões de jogo de fora.

Treina o TCN (por vídeo, 2 saídas, bola) deixando cada par de fora, 3 sementes, e guarda a média das
probabilidades em outputs/modelos/probs_logo.npz. Depois varre limiar x margem pré com a segmentação
real e mostra, por modalidade, o início preservado (média e pior par) e o tamanho da saída em relação
ao gabarito. Resultado em outputs/modelos/calibracao.csv.

Uso:
    uv run python scripts/calibrar_limiar.py [--reusar]
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from rallycut.bola import com_bola
from rallycut.config import carregar_config
from rallycut.deteccao import avaliar, segmentar
from rallycut.features import JANELA_S, sufixo
from rallycut.pares import listar_pares, preparar_par
from rallycut.temporal import padronizar, prever, treinar_tcn
from rallycut.treino import carregar, dados_tcn

SAIDA = Path("outputs/modelos")
LIMIARES = np.round(np.arange(0.05, 0.96, 0.05), 2)
MARGENS_PRE = (4.0, 6.0, 8.0)


def previsoes_logo(sementes: int) -> dict[str, np.ndarray]:
    altura = carregar_config("quadra")["features"]["altura"]
    pastas = [preparar_par(p, Path("outputs/pares"), altura=altura) for p in listar_pares()]
    d, _ = carregar(pastas, 32, sufixo(altura))
    pares, _, Xs, ys = dados_tcn(com_bola(d))
    probs = {}
    for i, par in enumerate(pares):
        treino = [j for j in range(len(pares)) if j != i]
        Z = padronizar(Xs, "video", treino)
        probs[par] = np.mean([prever(treinar_tcn([Z[j] for j in treino], [ys[j] for j in treino], semente=s), Z[i]).max(axis=1)
                              for s in range(sementes)], axis=0)
        print(f"{par}: ok", flush=True)
    np.savez(SAIDA / "probs_logo.npz", **probs)
    return probs


def varrer(probs: dict[str, np.ndarray]) -> pd.DataFrame:
    linhas = []
    for par, p in probs.items():
        mod = par.split("_")[0]
        trechos = pd.read_csv(Path("outputs/pares") / par / "trechos.csv")
        gabarito_min = trechos["duracao_s"].sum() / 60
        for margem in MARGENS_PRE:
            for lim in LIMIARES:
                cfg = {**carregar_config(mod)["deteccao"], "limiar": lim, "margem_pre_s": margem}
                r = avaliar(trechos, segmentar(p, cfg, len(p) * JANELA_S))
                linhas.append({"par": par, "modalidade": mod, "margem_pre_s": margem, "limiar": lim,
                               "inicios": r["inicios_preservados"], "razao": r["mantido_min"] / gabarito_min,
                               "falsos": r["falsos"], "morto": r["morto_residual"]})
    return pd.DataFrame(linhas)


def principal():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--reusar", action="store_true", help="usa outputs/modelos/probs_logo.npz já salvo")
    ap.add_argument("--sementes", type=int, default=3)
    args = ap.parse_args()

    probs = dict(np.load(SAIDA / "probs_logo.npz")) if args.reusar else previsoes_logo(args.sementes)
    r = varrer(probs)
    r.to_csv(SAIDA / "calibracao.csv", index=False)
    agg = r.groupby(["modalidade", "margem_pre_s", "limiar"]).agg(
        inicios=("inicios", "mean"), pior=("inicios", "min"), razao=("razao", "mean")).reset_index()
    for mod, g in agg.groupby("modalidade"):
        print(f"\n== {mod}: início preservado (média | pior par) e saída / gabarito ==")
        for margem, h in g.groupby("margem_pre_s"):
            h = h[(h["inicios"] >= 0.85) & (h["limiar"] <= 0.6)]
            print(f"margem pré {margem:.0f} s: " + " ".join(
                f"[{x.limiar:.2f}: {x.inicios:.0%}|{x.pior:.0%} {x.razao:.1f}x]" for x in h.itertuples()))


if __name__ == "__main__":
    principal()
