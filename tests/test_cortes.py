import cv2
import numpy as np
import pytest
import soundfile as sf

from rallycut.config import carregar_config
from rallycut.cortes import (detectar_cortes, extrair_diferencas, fluxo_audio,
                             trechos_entre_cortes, z_visual)

FPS = 30
CORTE_FRAME = 90


@pytest.fixture(scope="module")
def cfg():
    return carregar_config("quadra")["cortes"]


def _gerar_video(caminho, total=180, corte=CORTE_FRAME):
    """Câmera fixa: fundo texturizado + 6 "jogadores" que andam 1 px/frame.
    No frame `corte` todos os jogadores saltam para outras posições."""
    rng = np.random.default_rng(0)
    fundo = rng.integers(80, 120, (180, 320), dtype=np.uint8)
    pos_a = rng.integers(20, 100, (6, 2)) + [0, 20]
    pos_b = rng.integers(20, 100, (6, 2)) + [0, 20]
    escritor = cv2.VideoWriter(str(caminho), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (320, 180), False)
    for i in range(total):
        quadro = fundo.copy()
        pos = pos_a if i < corte else pos_b
        for x, y in pos:
            x0 = int(x + i)
            quadro[y:y + 14, x0:x0 + 14] = 255
        escritor.write(quadro)
    escritor.release()


def test_z_visual_destaca_pico_isolado(cfg):
    rng = np.random.default_rng(1)
    medias = rng.normal(0.003, 0.0003, 600).astype(np.float32)
    fracoes = rng.normal(0.004, 0.0004, 600).astype(np.float32)
    medias[300] = fracoes[300] = 0.05
    z = z_visual(medias, fracoes, 30, cfg)
    assert z.argmax() == 300
    assert z[300] > cfg["visual_alta"]
    assert np.delete(z, 300).max() < cfg["visual_min"]


def test_fluxo_audio_detecta_mudanca_de_espectro(tmp_path, cfg):
    rng = np.random.default_rng(2)
    taxa = 16000
    t = np.arange(5 * taxa) / taxa
    antes = 0.1 * rng.standard_normal(5 * taxa)
    depois = 0.3 * np.sin(2 * np.pi * 1500 * t) + 0.02 * rng.standard_normal(5 * taxa)
    wav = tmp_path / "a.wav"
    sf.write(wav, np.concatenate([antes, depois]).astype(np.float32), taxa)
    tempos, z = fluxo_audio(wav, cfg)
    assert abs(tempos[z.argmax()] - 5.0) < 0.05
    assert z.max() > cfg["audio_confirma"]
    assert np.sort(z)[-2] < z.max()  # pico único, não um platô


def test_corte_sintetico_e_encontrado(tmp_path, cfg):
    video = tmp_path / "v.mp4"
    _gerar_video(video)
    medias, fracoes = extrair_diferencas(video, nvdec=False)
    assert len(medias) == 179
    z = z_visual(medias, fracoes, FPS, cfg)
    cortes = detectar_cortes(z, FPS, np.zeros(0), np.zeros(0), cfg)
    assert len(cortes) == 1
    assert cortes.loc[0, "frame"] == CORTE_FRAME
    assert cortes.loc[0, "origem"] == "visual"


def test_piso_absoluto_descarta_movimento_em_cena_estatica(cfg):
    z = np.zeros(600)
    z[100] = z[300] = 30.0
    fracoes = np.full(600, 0.001)
    fracoes[100] = 0.06   # corte real: muitos pixels mudaram
    fracoes[300] = 0.008  # z alto, mas só um jogador se mexeu
    cortes = detectar_cortes(z, 30, np.zeros(0), np.zeros(0), cfg, fracoes)
    assert list(cortes["frame"]) == [101]


def test_trechos_entre_cortes():
    t = trechos_entre_cortes(np.array([10.0, 4.0]), 30.0)
    assert list(t["inicio_s"]) == [0.0, 4.0, 10.0]
    assert list(t["fim_s"]) == [4.0, 10.0, 30.0]
    assert list(t["rally"]) == [1, 2, 3]
