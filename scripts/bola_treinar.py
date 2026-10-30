"""Treina o detector de bola (mapa de calor) deixando um jogo de fora por vez (Portão 1b).

Usa os cliques de scripts/bola_rotular.py. Por jogo de fora e limiar de score: recall em frames de
rally com bola visível, falsos por frame e fração de frames com detecção (rally x tempo morto).
Critério do Portão 1b: recall >= 70% em rally com < 0,3 falso por frame, em cada jogo de fora.
Resultado em outputs/bola/mapa_logo.csv.

Uso:
    uv run python scripts/bola_treinar.py [--passos 1200] [--raio 25] [--jogos praia_set1 ...]
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from rallycut.bola import LARGURA, candidatos_bola, carregar_rotulos, treinar_mapa

SAIDA = Path("outputs/bola")
LIMIARES = (0.1, 0.2, 0.3, 0.5, 0.7)
ESCALA = LARGURA / 1920


def metricas(d: pd.DataFrame, cands: list[np.ndarray], raio: float) -> list[dict]:
    """Uma linha por limiar. Acerto: pico a <= `raio` px (escala 1080p) do clique."""
    linhas = []
    for lim in LIMIARES:
        acerto, fp, algum = [], [], []
        for c, r in zip(cands, d.itertuples()):
            c = c[c[:, 2] >= lim]
            h = bool(r.x == r.x and len(c) and (np.hypot(c[:, 0] / ESCALA - r.x, c[:, 1] / ESCALA - r.y) <= raio).any())
            acerto.append(h)
            fp.append(len(c) - h)
            algum.append(len(c) > 0)
        acerto, fp, algum = map(np.array, (acerto, fp, algum))
        rally, morto, vis = (d.classe == "rally").to_numpy(), (d.classe == "morto").to_numpy(), d["x"].notna().to_numpy()
        linhas.append({"limiar": lim, "recall_rally": acerto[rally & vis].mean(), "recall_morto": acerto[morto & vis].mean(),
                       "fp_por_frame": fp.mean(), "det_rally": algum[rally].mean(), "det_morto": algum[morto].mean()})
    return linhas


def principal():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--passos", type=int, default=1200)
    ap.add_argument("--raio", type=float, default=25.0)
    ap.add_argument("--jogos", nargs="+", default=None, help="jogos de fora a avaliar (padrão: todos)")
    args = ap.parse_args()

    d, imgs, centros = carregar_rotulos(SAIDA)
    resultados = []
    for jogo in args.jogos or sorted(d["par"].unique()):
        fora = (d["par"] == jogo).to_numpy()
        treino = np.flatnonzero(~fora)
        modelo = treinar_mapa(imgs[treino], [centros[i] for i in treino], args.passos)
        cands = candidatos_bola(modelo, imgs[fora])
        r = pd.DataFrame(metricas(d[fora].reset_index(drop=True), cands, args.raio)).assign(jogo=jogo)
        resultados.append(r)
        print(f"\n== {jogo} de fora ==\n{r.drop(columns='jogo').round(2).to_string(index=False)}", flush=True)
    todos = pd.concat(resultados)
    todos.to_csv(SAIDA / "mapa_logo.csv", index=False)
    print("\n== resumo por limiar (média dos jogos de fora; pior jogo entre parênteses) ==")
    for lim, g in todos.groupby("limiar"):
        print(f"limiar {lim:.1f}: recall rally {g.recall_rally.mean():.0%} ({g.recall_rally.min():.0%}) | "
              f"falsos/frame {g.fp_por_frame.mean():.2f} ({g.fp_por_frame.max():.2f}) | "
              f"detecção rally {g.det_rally.mean():.0%} x morto {g.det_morto.mean():.0%}")


if __name__ == "__main__":
    principal()
