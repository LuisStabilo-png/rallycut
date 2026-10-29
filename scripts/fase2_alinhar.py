"""Alinha o vídeo editado ao bruto e grava os trechos (rallies) e o tempo morto removido.

Saída em --saida: trechos.csv/json, removidos.csv e amostra.jpg (frame do editado ao
lado do frame do bruto no mesmo ponto: devem ser idênticos).

Uso:
    uv run python scripts/fase2_alinhar.py --bruto <bruto.mp4> --editado <editado.mp4> \
        --saida outputs/pares/quadra_set1 --modalidade quadra
"""

import argparse
import subprocess
import time
from pathlib import Path

import cv2
import numpy as np

from rallycut.alinhamento import alinhar_imagem, intervalos_removidos, refinar_audio
from rallycut.config import MODALIDADES, carregar_config
from rallycut.media import cmd_audio, ler_frames_cinza, sondar_video
from rallycut.relatorio import rotular


def _frame(video: Path, t: float, fps: float, largura: int = 480) -> np.ndarray:
    cap = cv2.VideoCapture(str(video))
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(t * fps))
    ok, img = cap.read()
    cap.release()
    return cv2.resize(img, (largura, round(img.shape[0] * largura / img.shape[1]))) if ok else None


def amostra(trechos, bruto, editado, fps_b, fps_e, destino: Path, n: int = 8):
    """Frame do editado (esq.) e do bruto (dir.) no meio de n trechos sorteados."""
    linhas = []
    for r in trechos.sample(min(n, len(trechos)), random_state=0).sort_index().itertuples():
        meio = (r.editado_fim - r.editado_ini) / 2
        a, b = _frame(editado, r.editado_ini + meio, fps_e), _frame(bruto, r.bruto_ini + meio, fps_b)
        if a is not None and b is not None:
            linhas.append(rotular(np.hstack([a, b]), f"trecho {r.Index + 1}  editado | bruto"))
    cv2.imwrite(str(destino), np.vstack(linhas), [cv2.IMWRITE_JPEG_QUALITY, 80])


def principal():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bruto", type=Path, required=True)
    ap.add_argument("--editado", type=Path, required=True)
    ap.add_argument("--saida", type=Path, required=True)
    ap.add_argument("--modalidade", choices=MODALIDADES, default="quadra")
    args = ap.parse_args()

    cfg = carregar_config(args.modalidade)
    c = cfg["alinhamento"]
    args.saida.mkdir(parents=True, exist_ok=True)
    info_b, info_e = sondar_video(args.bruto), sondar_video(args.editado)

    t0 = time.perf_counter()
    frames = {k: ler_frames_cinza(v, c["fps"], c["largura"], c["altura"])
              for k, v in (("editado", args.editado), ("bruto", args.bruto))}
    print(f"frames: {time.perf_counter() - t0:.0f}s ({len(frames['editado'])} editado, {len(frames['bruto'])} bruto)")
    trechos = alinhar_imagem(frames["editado"], frames["bruto"], c["fps"], c["salto"])
    wavs = {}
    for k, v in (("editado", args.editado), ("bruto", args.bruto)):
        wavs[k] = args.saida / f"audio_{k}.wav"
        subprocess.run(cmd_audio(v, wavs[k], cfg), check=True)
    trechos = refinar_audio(trechos, wavs["editado"], wavs["bruto"], c["janela_audio_s"])
    trechos.insert(0, "rally", range(1, len(trechos) + 1))
    trechos["duracao_s"] = trechos["editado_fim"] - trechos["editado_ini"]

    removidos = intervalos_removidos(trechos, info_b["duracao_s"])
    trechos.round(3).to_csv(args.saida / "trechos.csv", index=False)
    trechos.round(3).to_json(args.saida / "trechos.json", orient="records", indent=1)
    removidos.round(3).to_csv(args.saida / "removidos.csv", index=False)
    amostra(trechos, args.bruto, args.editado, info_b["fps"], info_e["fps"], args.saida / "amostra.jpg")

    d = trechos["duracao_s"]
    print(f"{len(trechos)} trechos | duração mediana {d.median():.1f}s, min {d.min():.1f}s, máx {d.max():.1f}s")
    print(f"mantido {d.sum() / 60:.1f} de {info_b['duracao_s'] / 60:.1f} min "
          f"({d.sum() / info_b['duracao_s']:.0%}) | editado tem {info_e['duracao_s'] / 60:.1f} min")
    print(f"custo médio por trecho: mediana {trechos['custo'].median():.2f}, máx {trechos['custo'].max():.2f}")
    print(f"ajuste de áudio: |mediana| {trechos['ajuste_s'].abs().median():.3f}s, "
          f"máx {trechos['ajuste_s'].abs().max():.2f}s | pico mediano {trechos['pico'].median():.0f}")


if __name__ == "__main__":
    principal()
