import numpy as np
import pandas as pd

from rallycut.deteccao import avaliar, com_contexto, segmentar
from rallycut.features import JANELA_S, rotular_janelas

CFG = {"limiar": 0.5, "suavizar_s": 1.0, "uniao_gap_s": 3.0, "duracao_min_s": 3.0,
       "margem_pre_s": 1.0, "margem_pos_s": 1.0}


def _prob(rallies, duracao_s=100):
    p = np.zeros(int(duracao_s / JANELA_S))
    for a, b in rallies:
        p[int(a / JANELA_S):int(b / JANELA_S)] = 0.9
    return p


def test_segmentar_une_gaps_curtos_descarta_curtos_e_poe_margens():
    p = _prob([(10, 20), (22, 30), (50, 52), (70, 80)])  # 20-22 s: gap curto; 50-52 s: curto demais
    segs = segmentar(p, CFG, 100)
    assert np.allclose(segs, [[9, 31], [69, 81]])


def test_avaliar_inicio_preservado_perdidos_e_falsos():
    gt = pd.DataFrame({"bruto_ini": [10.0, 40.0, 70.0], "bruto_fim": [20.0, 50.0, 80.0]})
    segs = np.array([[9.0, 21.0], [41.0, 49.0], [90.0, 95.0]])  # 40 s cortado em 1 s; 70 s perdido
    r = avaliar(gt, segs)
    assert r["inicios_preservados"] == 1 / 3
    assert r["perdidos"] == 1 and r["falsos"] == 1
    assert 0 < r["morto_residual"] < 1


def test_rotular_janelas():
    t = np.arange(0, 20, JANELA_S)
    r = rotular_janelas(t, pd.DataFrame({"bruto_ini": [5.0], "bruto_fim": [10.0]}))
    assert r[int(5 / JANELA_S)] == 1 and r[int(10 / JANELA_S) - 1] == 1 and r[0] == 0 and r[-1] == 0


def test_com_contexto_adiciona_colunas_sem_perder_linhas():
    df = pd.DataFrame(np.random.default_rng(0).random((50, 9)),
                      columns=["mov_med", "mov_frac", "n", "perto_med", "custo_hung", "rms_db", "fluxo",
                               "disp_x", "disp_y"])
    c = com_contexto(df)
    assert len(c) == 50 and c.shape[1] == 9 + 9 * 2 * 2
