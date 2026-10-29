"""Alinha um vídeo editado (só trechos removidos, mesma ordem) ao bruto.

Resultado: lista de trechos (início/fim no editado e no bruto). Os intervalos do bruto
que ficaram de fora são o tempo morto real, ou seja, o gabarito de cortes.

Método: com câmera fixa, cada frame do editado reaparece no bruto. Compara-se os frames
sem o fundo (sobram jogadores e bola) e resolve-se por Viterbi com tempo sempre crescente:
dentro de um trecho o bruto avança 1 frame por frame do editado; um salto custa `salto`.
O tempo de cada trecho é depois refinado pela correlação do áudio.
"""

import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
import torch
from scipy.signal import fftconvolve

from rallycut.media import cmd_audio, ler_frames_cinza, sondar_video
from rallycut.relatorio import amostra_alinhamento


def descritores(frames: np.ndarray) -> torch.Tensor:
    """Frames sem o fundo fixo, sem brilho médio e com norma unitária (n, pixels)."""
    x = frames.reshape(len(frames), -1).astype(np.float32)
    x -= np.median(x, axis=0)
    x -= x.mean(axis=1, keepdims=True)
    norma = np.linalg.norm(x, axis=1, keepdims=True)
    return torch.from_numpy(x / np.maximum(norma, 0.3 * np.median(norma)))


def viterbi(custo: torch.Tensor, salto: float) -> np.ndarray:
    """Melhor caminho j[i] (frame do bruto de cada frame do editado), j crescente."""
    n, m = custo.shape
    dev = custo.device
    j = torch.arange(m, device=dev)
    inf = torch.full((2,), float("inf"), device=dev)
    ant = torch.empty((n, m), dtype=torch.int16 if m < 32768 else torch.int32, device=dev)
    v = custo[0].clone()
    for i in range(1, n):
        continua = torch.cat([inf[:1], v[:-1]])                  # veio de j-1
        melhor, onde = torch.cummin(v, 0)
        pula = torch.cat([inf, melhor[:-2]]) + salto             # veio de qualquer j' <= j-2
        usa_continua = continua <= pula
        ant[i] = torch.where(usa_continua, j - 1, torch.cat([j[:2], onde[:-2]])).to(ant.dtype)
        v = torch.where(usa_continua, continua, pula) + custo[i]
    ant = ant.cpu().numpy()
    caminho = np.empty(n, int)
    caminho[-1] = int(v.argmin())
    for i in range(n - 1, 0, -1):
        caminho[i - 1] = ant[i, caminho[i]]
    return caminho


def alinhar_imagem(frames_editado, frames_bruto, fps: float, salto: float, device: str = "cuda") -> pd.DataFrame:
    """Trechos (editado -> bruto) pela imagem, com resolução de 1/fps s."""
    de, db = descritores(frames_editado).to(device), descritores(frames_bruto).to(device)
    custo = 1 - de @ db.T
    j = viterbi(custo, salto)
    quebras = np.flatnonzero(np.diff(j) != 1) + 1
    ini, fim = np.r_[0, quebras], np.r_[quebras, len(j)]
    custo_i = custo[torch.arange(len(j)), torch.from_numpy(j).to(device)].cpu().numpy()
    return pd.DataFrame({
        "editado_ini": ini / fps, "editado_fim": fim / fps,
        "bruto_ini": j[ini] / fps, "bruto_fim": (j[fim - 1] + 1) / fps,
        "custo": [custo_i[a:b].mean() for a, b in zip(ini, fim)]})


def refinar_audio(trechos: pd.DataFrame, wav_editado, wav_bruto, janela_s: float = 0.4,
                  max_s: float = 12.0) -> pd.DataFrame:
    """Corrige o deslocamento de cada trecho (<= janela_s) por correlação do áudio.

    Acrescenta `ajuste_s` (correção aplicada) e `pico` (nitidez da correlação; baixo = incerto)."""
    ae, sr = sf.read(str(wav_editado), dtype="float32")
    ab, _ = sf.read(str(wav_bruto), dtype="float32")
    t = trechos.copy()
    ajustes, picos = [], []
    for r in t.itertuples():
        dur = min(r.editado_fim - r.editado_ini - 0.2, max_s)
        e = ae[int((r.editado_ini + 0.1) * sr):int((r.editado_ini + 0.1 + dur) * sr)]
        ini = int((r.bruto_ini + 0.1 - janela_s) * sr)
        w = ab[max(ini, 0):int((r.bruto_ini + 0.1 + dur + janela_s) * sr)]
        if len(e) < sr // 2 or len(w) < len(e):
            ajustes.append(0.0), picos.append(np.nan)
            continue
        xc = fftconvolve(w, e[::-1], mode="valid")
        k = int(np.argmax(xc))
        ajustes.append((k + max(ini, 0) - ini) / sr - janela_s)
        picos.append(float(xc[k] / (np.median(np.abs(xc)) + 1e-9)))
    t["ajuste_s"], t["pico"] = ajustes, picos
    for c in ("bruto_ini", "bruto_fim"):
        t[c] += t["ajuste_s"]
    return t


def alinhar_par(bruto: Path, editado: Path, saida: Path, cfg: dict) -> pd.DataFrame:
    """Alinha um par bruto/editado e grava em `saida`: trechos.csv/json, removidos.csv, amostra.jpg
    e os áudios (audio_bruto.wav, audio_editado.wav). Devolve os trechos."""
    c = cfg["alinhamento"]
    saida.mkdir(parents=True, exist_ok=True)
    frames = [ler_frames_cinza(v, c["fps"], c["largura"], c["altura"]) for v in (editado, bruto)]
    trechos = alinhar_imagem(*frames, c["fps"], c["salto"])
    wavs = {}
    for nome, video in (("editado", editado), ("bruto", bruto)):
        wavs[nome] = saida / f"audio_{nome}.wav"
        subprocess.run(cmd_audio(video, wavs[nome], cfg), check=True)
    trechos = refinar_audio(trechos, wavs["editado"], wavs["bruto"], c["janela_audio_s"])
    trechos.insert(0, "rally", range(1, len(trechos) + 1))
    trechos["duracao_s"] = trechos["editado_fim"] - trechos["editado_ini"]
    removidos = intervalos_removidos(trechos, sondar_video(bruto)["duracao_s"])
    trechos.round(3).to_csv(saida / "trechos.csv", index=False)
    trechos.round(3).to_json(saida / "trechos.json", orient="records", indent=1)
    removidos.round(3).to_csv(saida / "removidos.csv", index=False)
    amostra_alinhamento(trechos, bruto, editado, saida / "amostra.jpg")
    return trechos


def intervalos_removidos(trechos: pd.DataFrame, duracao_bruto_s: float) -> pd.DataFrame:
    """Partes do bruto fora de todos os trechos: o tempo morto removido pela edição."""
    ini = np.r_[0.0, trechos["bruto_fim"].to_numpy()]
    fim = np.r_[trechos["bruto_ini"].to_numpy(), duracao_bruto_s]
    d = pd.DataFrame({"bruto_ini": ini, "bruto_fim": fim})
    d["duracao_s"] = d["bruto_fim"] - d["bruto_ini"]
    return d[d["duracao_s"] > 0].reset_index(drop=True)
