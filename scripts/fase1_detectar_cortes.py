"""Fase 1: detecta os pontos de corte de vídeos já editados.

Para cada vídeo grava em `outputs/fase1/<nome>/`:
  cortes.csv   - candidatos (id, tempo, frame, escores, confiança, aceito)
  trechos.csv  - trechos entre os cortes aceitos (cada um é um rally)
  sinais.npz   - sinais brutos (diferenças de frame e fluxo de áudio), reaproveitáveis
                 com --reusar-sinais para ajustar limiares sem reler o vídeo

Uso:
    uv run python scripts/fase1_detectar_cortes.py data/editados/quadra/*.mp4
A modalidade é lida do nome da pasta pai (quadra/praia) ou de --modalidade.
"""

import argparse
import subprocess
import time
from pathlib import Path

import numpy as np

from rallycut.config import MODALIDADES, carregar_config
from rallycut.cortes import (detectar_cortes, extrair_diferencas, fluxo_audio,
                             trechos_entre_cortes, z_visual)
from rallycut.media import cmd_audio, sondar_video


def processar(video: Path, modalidade: str, saida_base: Path, config: Path | None,
              nvdec: bool, reusar: bool = False):
    cfg = carregar_config(modalidade, config)
    cc = cfg["cortes"]
    info = sondar_video(video)
    destino = saida_base / video.stem
    destino.mkdir(parents=True, exist_ok=True)
    print(f"\n== {video.name} ({modalidade}) {info['largura']}x{info['altura']} "
          f"{info['fps']:.2f}fps {info['duracao_s'] / 60:.1f} min")

    t0 = time.perf_counter()
    arq_sinais = destino / "sinais.npz"
    if reusar and arq_sinais.exists():
        s = np.load(arq_sinais)
        # z_visual e z_audio dependem de limiares da config, então recalculamos a partir dos brutos
        medias, fracoes = s["medias"], s["fracoes"]
        t_audio, z_aud = s["t_audio"], s["z_aud"]
    else:
        wav = destino / "audio.wav"
        subprocess.run(cmd_audio(video, wav, cfg), check=True)
        t_audio, z_aud = fluxo_audio(wav, cc)
        medias, fracoes = extrair_diferencas(video, cc["largura"], cc["altura"], nvdec)
        np.savez_compressed(arq_sinais, t_audio=t_audio, z_aud=z_aud, medias=medias, fracoes=fracoes)
    z_vis = z_visual(medias, fracoes, info["fps"], cc)
    t1 = time.perf_counter()

    cortes = detectar_cortes(z_vis, info["fps"], t_audio, z_aud, cc, fracoes)
    aceitos = cortes.loc[cortes["aceito"], "tempo_s"].to_numpy()
    trechos = trechos_entre_cortes(aceitos, info["duracao_s"])
    cortes.to_csv(destino / "cortes.csv", index=False)
    trechos.to_csv(destino / "trechos.csv", index=False)

    print(f"   tempo: {t1 - t0:.0f}s | frames: {len(medias) + 1}")
    print(f"   candidatos: {len(cortes)} -> {cortes['confianca'].value_counts().to_dict()} "
          f"| origem: {cortes['origem'].value_counts().to_dict()} | aceitos: {len(aceitos)}")
    d = trechos["duracao_s"]
    print(f"   trechos: {len(trechos)} | duração (s) mediana {d.median():.1f}, "
          f"p5 {d.quantile(.05):.1f}, p95 {d.quantile(.95):.1f}, máx {d.max():.1f}, mín {d.min():.1f}")
    return cortes, trechos


def principal():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("videos", nargs="+", type=Path)
    ap.add_argument("--modalidade", choices=MODALIDADES, default=None)
    ap.add_argument("--config", type=Path, default=None)
    ap.add_argument("--saida", type=Path, default=Path("outputs/fase1"))
    ap.add_argument("--sem-nvdec", action="store_true")
    ap.add_argument("--reusar-sinais", action="store_true",
                    help="usa o sinais.npz já salvo (não relê o vídeo); útil ao ajustar limiares")
    args = ap.parse_args()
    for video in args.videos:
        modalidade = args.modalidade or video.resolve().parent.name
        if modalidade not in MODALIDADES:
            ap.error(f"não deu para inferir a modalidade de {video}; use --modalidade")
        processar(video, modalidade, args.saida, args.config, not args.sem_nvdec, args.reusar_sinais)


if __name__ == "__main__":
    principal()
