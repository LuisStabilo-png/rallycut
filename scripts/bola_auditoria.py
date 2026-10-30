"""Portão 4: auditoria visual dos inícios de rally que o modelo (com bola) ainda perde.

Para cada jogo de fora: treina o TCN (por vídeo, 2 saídas, features de bola), média das sementes,
limiar que deixa o vídeo final com `--razao` x o gabarito (margem pré 4 s), lista os inícios perdidos
e salva em outputs/bola/auditoria_<par>.jpg os frames de 4 s antes, 2 s antes, no início e 2 s depois.

Uso:
    uv run python scripts/bola_auditoria.py --pares praia_set2 praia_set4 [--sementes 3]
"""

import argparse
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from rallycut.bola import com_bola
from rallycut.config import carregar_config
from rallycut.deteccao import segmentar
from rallycut.features import JANELA_S, sufixo
from rallycut.media import extrair_frame
from rallycut.pares import listar_pares, preparar_par
from rallycut.temporal import padronizar, prever, treinar_tcn
from rallycut.treino import carregar, dados_tcn, no_orcamento

SAIDA = Path("outputs/bola")


def principal():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pares", nargs="+", required=True)
    ap.add_argument("--sementes", type=int, default=3)
    ap.add_argument("--razao", type=float, default=1.5)
    ap.add_argument("--max-perdidos", type=int, default=8)
    args = ap.parse_args()

    altura = carregar_config("quadra")["features"]["altura"]
    pastas = [preparar_par(p, Path("outputs/pares"), altura=altura) for p in listar_pares()]
    d, _ = carregar(pastas, 32, sufixo(altura))
    pares, _, Xs, ys = dados_tcn(com_bola(d))
    for par in args.pares:
        i, mod = pares.index(par), par.split("_")[0]
        treino = [j for j in range(len(pares)) if j != i]
        Z = padronizar(Xs, "video", treino)
        p = np.mean([prever(treinar_tcn([Z[j] for j in treino], [ys[j] for j in treino], semente=s), Z[i]).max(axis=1)
                     for s in range(args.sementes)], axis=0)
        trechos = pd.read_csv(f"outputs/pares/{par}/trechos.csv")
        lim, r = no_orcamento(trechos, p, mod, args.razao)
        segs = segmentar(p, {**carregar_config(mod)["deteccao"], "margem_pre_s": 4.0, "limiar": lim}, len(p) * JANELA_S)
        perdidos = [(k, t) for k, t in zip(trechos["rally"], trechos["bruto_ini"])
                    if not any(a <= t <= b for a, b in segs)]
        print(f"\n{par}: limiar {lim:.2f} | início preservado {r['inicios_preservados']:.0%} | perdidos {r['perdidos']}/{r['rallies']} "
              f"| falsos {r['falsos']} | tempo morto sobrando {r['morto_residual']:.0%} | mantido {r['mantido_min']:.1f} min "
              f"(gabarito {trechos['duracao_s'].sum() / 60:.1f}) | inícios perdidos: "
              + ", ".join(f"#{k} @ {t:.0f}s" for k, t in perdidos))
        bruto, linhas = next(Path("data/pares", par).glob("*bruto*")), []
        for k, t in perdidos[:args.max_perdidos]:
            quadros = []
            for dt in (-4, -2, 0, 2):
                extrair_frame(bruto, max(t + dt, 0), SAIDA / "tmp.jpg", 640, 360)
                q = cv2.imread(str(SAIDA / "tmp.jpg"))
                quadros.append(cv2.putText(q, f"#{k} inicio{dt:+d}s", (6, 22), 0, 0.7, (0, 255, 255), 2))
            linhas.append(np.hstack(quadros))
        if linhas:
            cv2.imwrite(str(SAIDA / f"auditoria_{par}.jpg"), np.vstack(linhas))
            print(f"  frames dos inícios perdidos em {SAIDA / f'auditoria_{par}.jpg'}")


if __name__ == "__main__":
    principal()
