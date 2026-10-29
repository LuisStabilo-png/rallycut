"""Do vetor de probabilidades por janela aos rallies, e a avaliação contra o gabarito."""

import numpy as np
import pandas as pd

from rallycut.features import JANELA_S

BASE = ["mov_med", "mov_frac", "n", "perto_med", "custo_hung", "rms_db", "fluxo", "disp_x", "disp_y"]


def com_contexto(df: pd.DataFrame, janelas: tuple = (10, 30)) -> pd.DataFrame:
    """Acrescenta média e desvio móveis centrados (5 s e 15 s): uma janela sozinha é ambígua,
    o padrão ao redor (jogo contínuo x parada) é que diferencia rally de tempo morto."""
    extra = {f"{c}_{op}{k}": getattr(df[c].rolling(k, center=True, min_periods=1), op)()
             for c in BASE for k in janelas for op in ("mean", "std")}
    return pd.concat([df, pd.DataFrame(extra)], axis=1)


def segmentar(p: np.ndarray, cfg: dict, duracao_s: float) -> np.ndarray:
    """Probabilidades por janela -> rallies (n, 2) em segundos [início, fim].

    Suaviza, aplica o limiar, une paradas curtas, descarta trechos curtos e põe as margens."""
    # janela ímpar: simétrica, então a suavização não desloca o início nem o fim dos rallies
    s = pd.Series(p).rolling(int(cfg["suavizar_s"] / JANELA_S) | 1, center=True, min_periods=1).median()
    on = (s >= cfg["limiar"]).to_numpy()
    ini = np.flatnonzero(on & ~np.r_[False, on[:-1]]) * JANELA_S
    fim = (np.flatnonzero(on & ~np.r_[on[1:], False]) + 1) * JANELA_S
    uniao = []
    for a, b in zip(ini, fim):
        if uniao and a - uniao[-1][1] <= cfg["uniao_gap_s"]:
            uniao[-1][1] = b
        else:
            uniao.append([a, b])
    segs = np.array([[max(a - cfg["margem_pre_s"], 0), min(b + cfg["margem_pos_s"], duracao_s)]
                     for a, b in uniao if b - a >= cfg["duracao_min_s"]]).reshape(-1, 2)
    return segs


def avaliar(gabarito: pd.DataFrame, segs: np.ndarray) -> dict:
    """Compara os rallies detectados com os trechos do editado (colunas bruto_ini/bruto_fim).

    - inicios_preservados: fração de rallies do gabarito cujo início está dentro de um detectado
      (o início detectado é <= ao real: nunca cortou o saque);
    - perdidos: rallies do gabarito com menos de metade coberta;
    - falsos: detectados que não tocam nenhum rally do gabarito;
    - morto_residual: parte do que foi mantido que não é rally (tempo morto sobrando)."""
    gt = gabarito[["bruto_ini", "bruto_fim"]].to_numpy()

    def sobreposicao(a, b):
        return max(0.0, min(a[1], b[1]) - max(a[0], b[0]))

    cobertura = [sum(sobreposicao(g, s) for s in segs) / (g[1] - g[0]) for g in gt]
    mantido = float((segs[:, 1] - segs[:, 0]).sum())
    dentro = sum(sobreposicao(g, s) for g in gt for s in segs)
    return {
        "rallies": len(gt), "detectados": len(segs),
        "inicios_preservados": float(np.mean([any(s[0] <= g[0] <= s[1] for s in segs) for g in gt])),
        "perdidos": int(sum(c < 0.5 for c in cobertura)),
        "falsos": int(sum(not any(sobreposicao(g, s) > 0 for g in gt) for s in segs)),
        "morto_residual": 1 - dentro / mantido if mantido else 1.0,
        "mantido_min": mantido / 60,
    }
