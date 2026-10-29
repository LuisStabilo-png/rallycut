import numpy as np
import pandas as pd
import soundfile as sf

from rallycut.alinhamento import alinhar_imagem, intervalos_removidos, refinar_audio

FPS = 10


def test_alinhar_imagem_acha_trechos_e_saltos():
    rng = np.random.default_rng(0)
    bruto = (128 + np.cumsum(rng.normal(0, 4, (600, 36 * 24)), axis=0)).reshape(600, 24, 36)
    partes = [(100, 250), (320, 400), (500, 580)]
    editado = np.concatenate([bruto[a:b] for a, b in partes]) * 1.1 + rng.normal(0, 2, (310, 24, 36))
    t = alinhar_imagem(editado, bruto, FPS, salto=3.0, device="cpu")
    assert len(t) == 3
    assert np.allclose(t["bruto_ini"], [10, 32, 50], atol=0.2)
    assert np.allclose(t["bruto_fim"], [25, 40, 58], atol=0.2)
    assert np.allclose(t["editado_ini"], [0, 15, 23], atol=0.2)


def test_refinar_audio_corrige_deslocamento(tmp_path):
    rng = np.random.default_rng(1)
    sr = 16000
    bruto = rng.normal(0, 0.1, 40 * sr).astype(np.float32)
    editado = bruto[10 * sr:20 * sr]  # trecho que começa em 10 s no bruto
    sf.write(tmp_path / "b.wav", bruto, sr)
    sf.write(tmp_path / "e.wav", editado, sr)
    t = pd.DataFrame({"editado_ini": [0.0], "editado_fim": [10.0], "bruto_ini": [10.2], "bruto_fim": [20.2]})
    r = refinar_audio(t, tmp_path / "e.wav", tmp_path / "b.wav")
    assert abs(r.loc[0, "bruto_ini"] - 10.0) < 0.01
    assert abs(r.loc[0, "ajuste_s"] + 0.2) < 0.01


def test_intervalos_removidos():
    t = pd.DataFrame({"bruto_ini": [10.0, 30.0], "bruto_fim": [20.0, 35.0]})
    d = intervalos_removidos(t, 40.0)
    assert list(d["bruto_ini"]) == [0.0, 20.0, 35.0]
    assert list(d["duracao_s"]) == [10.0, 10.0, 5.0]
