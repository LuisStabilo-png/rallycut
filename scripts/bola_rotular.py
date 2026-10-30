"""Rotulador de bola: clique no CENTRO da bola em cada frame de outputs/bola/amostra.csv.

Teclas:  clique = marca a bola e avança | n = sem bola visível e avança | b = volta | q = salva e sai
A ordem é embaralhada (semente fixa): se parar no meio, o que foi rotulado continua equilibrado.
O progresso fica em outputs/bola/rotulos.csv; ao reabrir, continua de onde parou.
Coordenadas salvas na escala do frame (1920x1080).

Uso:
    uv run python scripts/bola_rotular.py
"""

from pathlib import Path

import cv2
import numpy as np
import pandas as pd

SAIDA = Path("outputs/bola")
LARGURA_JANELA = 1600


def principal():
    amostra = pd.read_csv(SAIDA / "amostra.csv")
    ordem = np.random.default_rng(1).permutation(len(amostra))
    csv = SAIDA / "rotulos.csv"
    rotulos = {r.arquivo: (r.x, r.y) for r in pd.read_csv(csv).itertuples()} if csv.exists() else {}
    esc = LARGURA_JANELA / 1920
    clique = {}

    def ao_clicar(evento, x, y, *_):
        if evento == cv2.EVENT_LBUTTONDOWN:
            clique["p"] = (x / esc, y / esc)

    def salvar():
        pd.DataFrame([(a, x, y) for a, (x, y) in rotulos.items()], columns=["arquivo", "x", "y"]).to_csv(csv, index=False)

    cv2.namedWindow("bola", cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback("bola", ao_clicar)
    k = 0
    while 0 <= k < len(ordem):
        arq = amostra["arquivo"].iloc[ordem[k]]
        if arq in rotulos and "pular" not in clique:
            k += 1
            continue
        clique.clear()
        img = cv2.resize(cv2.imread(str(SAIDA / "frames" / arq)), None, fx=esc, fy=esc, interpolation=cv2.INTER_AREA)
        cv2.putText(img, f"{len(rotulos)}/{len(ordem)}  clique=bola  n=sem bola  b=volta  q=sair", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.imshow("bola", img)
        while "p" not in clique:
            tecla = cv2.waitKey(20) & 0xFF
            if tecla == ord("n"):
                clique["p"] = (np.nan, np.nan)
            elif tecla == ord("b"):
                k = max(k - 1, 0)
                clique["pular"] = True
                break
            elif tecla == ord("q"):
                salvar()
                return
        if "p" in clique:
            rotulos[arq] = clique["p"]
            k += 1
            if len(rotulos) % 10 == 0:
                salvar()
    salvar()
    print(f"{len(rotulos)} frames rotulados em {csv}")


if __name__ == "__main__":
    principal()
