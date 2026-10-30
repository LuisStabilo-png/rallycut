"""Mede o detector de bola contra os cliques de scripts/bola_rotular.py (Portão 1).

Acerto: o clique cai dentro da caixa detectada (com folga de `--folga` px, na escala 1080p).
Por limiar de score, em frames com bola visível: recall (por classe e por par); em todos os frames:
falsos positivos por frame e fração de frames com alguma detecção (rally x tempo morto, que já é
o sinal de "bola presente"). Resultado também em outputs/bola/avaliacao_<altura>.csv.

Uso:
    uv run python scripts/bola_avaliar.py [--alturas 720 1080]
"""

import argparse
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from rallycut.bola import carregar_detector_bola, detectar_bolas

SAIDA = Path("outputs/bola")
LIMIARES = (0.05, 0.1, 0.2, 0.3, 0.5, 0.7)


def detectar(altura: int, arquivos: list[str], lote: int = 4) -> tuple[list[np.ndarray], float]:
    """Caixas em pixels (1920x1080) por frame e segundos por frame."""
    modelo = carregar_detector_bola(altura)
    caixas, t0 = [], time.time()
    for i in range(0, len(arquivos), lote):
        imgs = [cv2.resize(cv2.imread(str(SAIDA / "frames" / a)), (round(altura * 16 / 9), altura),
                           interpolation=cv2.INTER_AREA) for a in arquivos[i:i + lote]]
        for c in detectar_bolas(modelo, imgs):
            caixas.append(c * np.array([1920, 1080, 1920, 1080, 1]))
    return caixas, (time.time() - t0) / len(arquivos)


def acertou(caixas: np.ndarray, x: float, y: float, folga: float) -> bool:
    return bool(len(caixas)) and bool(((caixas[:, 0] - folga <= x) & (x <= caixas[:, 2] + folga)
                                       & (caixas[:, 1] - folga <= y) & (y <= caixas[:, 3] + folga)).any())


def principal():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--alturas", type=int, nargs="+", default=[720, 1080])
    ap.add_argument("--folga", type=float, default=10.0)
    args = ap.parse_args()

    d = pd.read_csv(SAIDA / "amostra.csv").merge(pd.read_csv(SAIDA / "rotulos.csv"), on="arquivo")
    d["visivel"] = d["x"].notna()
    print(f"{len(d)} frames rotulados; bola visível em "
          f"{d.loc[d.classe == 'rally', 'visivel'].mean():.0%} dos de rally e "
          f"{d.loc[d.classe == 'morto', 'visivel'].mean():.0%} dos de tempo morto\n")
    for altura in args.alturas:
        caixas, s_frame = detectar(altura, d["arquivo"].tolist())
        linhas = []
        for lim in LIMIARES:
            dets = [c[c[:, 4] >= lim] for c in caixas]
            hit = np.array([r.visivel and acertou(c, r.x, r.y, args.folga) for c, r in zip(dets, d.itertuples())])
            fp = np.array([len(c) - (1 if h else 0) for c, h in zip(dets, hit)])
            algum = np.array([len(c) > 0 for c in dets])
            larg = [np.median(c[:, 2] - c[:, 0]) for c, h in zip(dets, hit) if h]
            L = {"limiar": lim, "tam_med_px": np.median(larg) if larg else np.nan, "fp_por_frame": fp.mean(),
                 "det_rally": algum[d.classe == "rally"].mean(), "det_morto": algum[d.classe == "morto"].mean(),
                 "fp_morto": fp[d.classe == "morto"].mean()}
            for classe in ("rally", "morto"):
                m = (d.classe == classe) & d.visivel
                L[f"recall_{classe}"] = hit[m].mean() if m.any() else np.nan
            for par, g in d[d.visivel & (d.classe == "rally")].groupby("par"):
                L[f"recall_rally_{par}"] = hit[g.index].mean()
            linhas.append(L)
        r = pd.DataFrame(linhas)
        r.to_csv(SAIDA / f"avaliacao_{altura}.csv", index=False)
        print(f"== {altura}p: {s_frame * 1000:.0f} ms/frame ==")
        print(r.round(2).to_string(index=False), "\n")


if __name__ == "__main__":
    principal()
