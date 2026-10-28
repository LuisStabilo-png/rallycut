"""Detecção dos pontos de corte em vídeos já editados (câmera fixa).

Num vídeo de câmera fixa, frames consecutivos são quase idênticos. Um corte seco
faz os jogadores "teletransportarem" e o espectro do áudio saltar. Combinamos:

  - sinal visual: diferença entre frames consecutivos em baixa resolução,
    normalizada por mediana/MAD local (z robusto), para um ataque rápido não
    parecer corte;
  - sinal de áudio: fluxo espectral entre janelas curtas, também em z robusto.

Cada corte sai com uma confiança (alta, media, baixa) para guiar a validação.
"""

import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
from scipy.ndimage import median_filter
from scipy.signal import find_peaks

LIMIAR_PIXEL_MUDOU = 20 / 255  # variação mínima para um pixel contar como "mudou"


def extrair_diferencas(video: Path, largura: int = 256, altura: int = 144,
                       nvdec: bool = True, bloco: int = 256) -> tuple[np.ndarray, np.ndarray]:
    """Lê o vídeo em tons de cinza e devolve, por par de frames consecutivos,
    (diferença média, fração de pixels que mudaram). O item k compara o frame k
    com o frame k+1, então o corte entre eles ocorre no tempo (k+1)/fps."""
    pre = ["-hwaccel", "cuda", "-hwaccel_output_format", "cuda"] if nvdec else []
    filtro = (f"scale_cuda={largura}:{altura},hwdownload,format=nv12" if nvdec
              else f"scale={largura}:{altura}")
    cmd = ["ffmpeg", "-nostdin", "-loglevel", "error", *pre, "-i", str(video), "-an",
           "-vf", filtro, "-pix_fmt", "gray", "-f", "rawvideo", "-"]
    tam = largura * altura
    medias, fracoes = [], []
    anterior = None
    with tempfile.TemporaryFile() as erros:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=erros)
        while True:
            dados = proc.stdout.read(tam * bloco)
            n = len(dados) // tam
            if n == 0:
                break
            frames = np.frombuffer(dados, np.uint8, n * tam).reshape(n, tam).astype(np.int16)
            if anterior is not None:
                frames = np.vstack([anterior, frames])
            if len(frames) > 1:
                dif = np.abs(np.diff(frames, axis=0)).astype(np.float32) / 255
                medias.append(dif.mean(axis=1))
                fracoes.append((dif > LIMIAR_PIXEL_MUDOU).mean(axis=1))
            anterior = frames[-1:]
        proc.stdout.close()
        if proc.wait() != 0:
            erros.seek(0)
            raise RuntimeError(f"ffmpeg falhou:\n{erros.read().decode(errors='replace')[-2000:]}")
    if not medias:
        return np.zeros(0, np.float32), np.zeros(0, np.float32)
    return np.concatenate(medias), np.concatenate(fracoes)


def z_robusto(x: np.ndarray, janela: int, piso: float) -> np.ndarray:
    """(x - mediana local) / (1,4826 * MAD local + piso). O piso evita z enorme em trechos estáticos."""
    janela = max(3, janela | 1)
    med = median_filter(x, size=janela, mode="nearest")
    mad = median_filter(np.abs(x - med), size=janela, mode="nearest")
    return (x - med) / (1.4826 * mad + piso)


def z_visual(medias: np.ndarray, fracoes: np.ndarray, fps: float, cfg: dict) -> np.ndarray:
    """Pontuação visual por par de frames: o maior z entre diferença média e fração alterada."""
    janela = int(cfg["janela_z_s"] * fps)
    piso = cfg["piso_visual"]
    return np.maximum(z_robusto(medias, janela, piso), z_robusto(fracoes, janela, piso))


def fluxo_audio(wav: Path, cfg: dict) -> tuple[np.ndarray, np.ndarray]:
    """Descontinuidade espectral do áudio: devolve (tempos em s, z robusto).

    Energia em bandas log-espaçadas por janela de 25 ms (passo de 10 ms); a
    distância entre janelas vizinhas, normalizada localmente, destaca saltos bruscos."""
    sinal, taxa = sf.read(str(wav), dtype="float32", always_2d=True)
    sinal = sinal.mean(axis=1)
    janela, passo = int(0.025 * taxa), int(0.010 * taxa)
    if len(sinal) < janela + passo:
        return np.zeros(0), np.zeros(0)
    n = 1 + (len(sinal) - janela) // passo
    idx = np.arange(janela)[None, :] + passo * np.arange(n)[:, None]
    espectro = np.abs(np.fft.rfft(sinal[idx] * np.hanning(janela), axis=1)) ** 2
    freqs = np.fft.rfftfreq(janela, 1 / taxa)
    bordas = np.geomspace(100, min(7500, taxa / 2 - 1), 41)
    bandas = np.stack([espectro[:, (freqs >= a) & (freqs < b)].sum(axis=1)
                       for a, b in zip(bordas[:-1], bordas[1:])], axis=1)
    log_e = np.log(bandas + 1e-10)
    # Compara a média de `w` janelas antes com a de `w` depois de cada ponto: a
    # média reduz o ruído aleatório do espectro e o pico cai exatamente na fronteira.
    w = int(cfg["janelas_audio"])
    soma = np.vstack([np.zeros((1, log_e.shape[1])), np.cumsum(log_e, axis=0)])
    k = np.arange(w, n - w + 1)  # fronteira entre as janelas k-1 e k
    antes = (soma[k] - soma[k - w]) / w
    depois = (soma[k + w] - soma[k]) / w
    dist = np.abs(depois - antes).mean(axis=1)
    tempos = (k * passo + janela / 2 - passo / 2) / taxa
    z = z_robusto(dist, int(cfg["janela_z_s"] / 0.010), cfg["piso_audio"])
    return tempos, z


def detectar_cortes(z_vis: np.ndarray, fps: float, t_audio: np.ndarray,
                    z_aud: np.ndarray, cfg: dict, fracoes: np.ndarray | None = None) -> pd.DataFrame:
    """Funde os dois sinais em uma tabela de cortes com confiança.

    O z é relativo ao ruído local e explode em cenas muito estáticas (qualquer
    jogador em movimento vira "corte"). Por isso, quando `fracoes` é dado, exigimos
    também que ao menos `fracao_min` dos pixels tenham mudado: um corte real troca a
    posição de todos os jogadores, um movimento normal não."""
    dist = max(1, int(cfg["distancia_min_s"] * fps))
    picos, _ = find_peaks(z_vis, height=cfg["visual_min"], distance=dist)
    if fracoes is not None:
        picos = picos[fracoes[picos] >= cfg["fracao_min"]]
    linhas = []
    for k in picos:
        t = (k + 1) / fps
        z_a = _max_audio(t_audio, z_aud, t, cfg["tolerancia_audio_s"])
        linhas.append({"tempo_s": t, "frame": int(k + 1), "z_visual": float(z_vis[k]),
                       "z_audio": z_a, "origem": "visual"})
    # Cortes que quase não mexem a imagem (jogadores parados) só aparecem no áudio.
    if len(z_aud) and cfg["audio_solo"] is not None:
        dist_a = max(1, int(cfg["distancia_min_s"] / 0.010))
        picos_a, _ = find_peaks(z_aud, height=cfg["audio_solo"], distance=dist_a)
        tempos_visuais = np.array([l["tempo_s"] for l in linhas])
        for k in picos_a:
            t = float(t_audio[k])
            if len(tempos_visuais) and np.min(np.abs(tempos_visuais - t)) < 2 * cfg["tolerancia_audio_s"]:
                continue
            frame = int(round(t * fps))
            linhas.append({"tempo_s": round(frame / fps, 4), "frame": frame,
                           "z_visual": float(z_vis[min(max(frame - 1, 0), len(z_vis) - 1)]) if len(z_vis) else 0.0,
                           "z_audio": float(z_aud[k]), "origem": "audio"})
    tabela = pd.DataFrame(linhas, columns=["tempo_s", "frame", "z_visual", "z_audio", "origem"])
    tabela = tabela.sort_values("tempo_s").reset_index(drop=True)
    tabela["confianca"] = [classificar(r, cfg) for r in tabela.itertuples()]
    tabela["aceito"] = tabela["confianca"] != "baixa"  # só estes dividem os trechos
    tabela.insert(0, "id", np.arange(1, len(tabela) + 1))
    return tabela


def _max_audio(tempos: np.ndarray, z: np.ndarray, t: float, tol: float) -> float:
    if not len(z):
        return 0.0
    ini, fim = np.searchsorted(tempos, [t - tol, t + tol])
    return float(z[ini:fim].max()) if fim > ini else 0.0


def classificar(corte, cfg: dict) -> str:
    audio_ok = corte.z_audio >= cfg["audio_confirma"]
    if corte.origem == "audio":
        return "baixa"
    if corte.z_visual >= cfg["visual_alta"] and audio_ok:
        return "alta"
    if corte.z_visual >= cfg["visual_alta"] or audio_ok:
        return "media"
    return "baixa"


def aplicar_correcoes(cortes: pd.DataFrame, correcoes: pd.DataFrame, fps: float) -> pd.DataFrame:
    """Aplica as correções feitas na validação manual.

    `correcoes` tem as colunas `acao` e `valor`:
      remover,<id>     - o candidato não é corte;
      aceitar,<id>     - o candidato (ex.: confiança baixa) é corte de verdade;
      adicionar,<s>    - corte perdido no tempo `s` (segundos)."""
    cortes = cortes.copy()
    for c in correcoes.itertuples():
        acao = str(c.acao).strip().lower()
        if acao in ("remover", "aceitar"):
            alvo = cortes["id"] == int(c.valor)
            if not alvo.any():
                raise ValueError(f"correção refere-se a um id inexistente: {c.valor}")
            cortes.loc[alvo, "aceito"] = acao == "aceitar"
            cortes.loc[alvo, "origem"] = "manual"
        elif acao == "adicionar":
            frame = int(round(float(c.valor) * fps))
            cortes.loc[len(cortes)] = {"id": cortes["id"].max() + 1, "tempo_s": frame / fps,
                                       "frame": frame, "z_visual": 0.0, "z_audio": 0.0,
                                       "origem": "manual", "confianca": "manual", "aceito": True}
        else:
            raise ValueError(f"ação desconhecida: {c.acao!r} (use remover, aceitar ou adicionar)")
    return cortes.sort_values("tempo_s").reset_index(drop=True)


def trechos_entre_cortes(tempos: np.ndarray, duracao_s: float) -> pd.DataFrame:
    """Cada trecho entre cortes consecutivos é um rally (positivo rotulado)."""
    bordas = np.concatenate([[0.0], np.sort(tempos), [duracao_s]])
    t = pd.DataFrame({"inicio_s": bordas[:-1], "fim_s": bordas[1:]})
    t["duracao_s"] = t["fim_s"] - t["inicio_s"]
    t.insert(0, "rally", np.arange(1, len(t) + 1))
    return t
