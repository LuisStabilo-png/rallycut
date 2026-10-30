import numpy as np
import pandas as pd

from rallycut.bola import ALTURA, LARGURA, features_bola


def test_features_bola_ignora_parada_e_mede_a_que_se_move():
    fps, n_janelas = 8, 100  # 50 s
    n = n_janelas * 4
    parada = pd.DataFrame({"i": np.arange(n), "x": 300.0, "y": 600.0, "score": 0.9})  # bola reserva na areia
    t = np.arange(100, 140)  # bola em jogo: 5 s, 40 px por frame
    movel = pd.DataFrame({"i": t, "x": 200.0 + 40 * (t - 100), "y": 200.0 + (t - 100), "score": 0.8})
    f = features_bola(pd.concat([parada, movel]), n_janelas, fps)
    com_bola = f.iloc[25:34]  # janelas 25..33 = frames 100..135
    assert (com_bola["bola_frac"] == 1).all() and (com_bola["bola_score"] == 0.8).all()
    assert np.allclose(com_bola["bola_vel"].iloc[1:], 40 * fps / LARGURA, rtol=0.01)
    assert (f.iloc[:20]["bola_frac"] == 0).all() and f.iloc[:20]["bola_vel"].isna().all()  # só a parada: ignorada
    assert (com_bola["bola_y"] < 400 / ALTURA).all()


def test_features_bola_sem_candidatos():
    f = features_bola(pd.DataFrame(columns=["i", "x", "y", "score"]), 10)
    assert len(f) == 10 and (f["bola_frac"] == 0).all() and f["bola_score"].isna().all()
