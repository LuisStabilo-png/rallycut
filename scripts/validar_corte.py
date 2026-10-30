"""Valida um corte contra o gabarito (pares do dataset) e confere o vídeo gerado.

Compara o rallies.csv do `cortar.py` com outputs/pares/<par>/trechos.csv (início preservado, perdidos,
falsos, tempo morto sobrando) e, com --video, confere com o ffprobe a duração (= soma dos rallies),
resolução, fps e presença de áudio.

Uso:
    uv run python scripts/validar_corte.py outputs/cortes/<nome>/rallies.csv <par> [--video cortado.mp4]
"""

import argparse
import json
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

from rallycut.deteccao import avaliar


def principal():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("rallies", type=Path)
    ap.add_argument("par")
    ap.add_argument("--video", type=Path, default=None)
    args = ap.parse_args()

    r = pd.read_csv(args.rallies)
    trechos = pd.read_csv(Path("outputs/pares") / args.par / "trechos.csv")
    m = avaliar(trechos, r[["ini_s", "fim_s"]].to_numpy())
    print(f"{args.par}: início preservado {m['inicios_preservados']:.0%} | perdidos {m['perdidos']}/{m['rallies']} | "
          f"falsos {m['falsos']} | tempo morto sobrando {m['morto_residual']:.0%} | "
          f"saída {m['mantido_min']:.1f} min = {m['mantido_min'] / (trechos['duracao_s'].sum() / 60):.2f}x o gabarito")
    if args.video:
        info = json.loads(subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "stream=codec_type,codec_name,width,height,r_frame_rate:format=duration",
             "-of", "json", str(args.video)], capture_output=True, text=True, check=True).stdout)
        esperado = float((r["fim_s"] - r["ini_s"]).sum())
        real = float(info["format"]["duration"])
        v = next(s for s in info["streams"] if s["codec_type"] == "video")
        print(f"vídeo: {real:.2f} s (esperado {esperado:.2f} s, dif {real - esperado:+.2f} s) | {v['width']}x{v['height']} "
              f"{v['r_frame_rate']} fps {v['codec_name']} | áudio: {'sim' if any(s['codec_type'] == 'audio' for s in info['streams']) else 'NÃO'}")


if __name__ == "__main__":
    principal()
