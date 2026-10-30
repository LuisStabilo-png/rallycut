"""Corte final: codifica cada rally (NVENC) e junta tudo num único vídeo."""

import subprocess
import tempfile
from pathlib import Path

import pandas as pd

from rallycut.media import cmd_saida, sondar_video


def cortar_video(bruto: Path, rallies: pd.DataFrame, destino: Path, cfg: dict, progresso=print) -> Path:
    """Um arquivo com os rallies (`ini_s`, `fim_s` em s do bruto) em sequência. Cada trecho é codificado
    com as mesmas configurações e juntado sem recodificar (concat). Se a GPU falhar, cai para a CPU."""
    info = sondar_video(bruto)
    destino.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=destino.parent) as tmp:
        tmp = Path(tmp)
        lista = []
        for r in rallies.itertuples():
            parte = tmp / f"parte_{r.rally:04d}.mp4"
            trecho = (max(r.ini_s, 0.0), min(r.fim_s, info["duracao_s"]))
            ok = subprocess.run(cmd_saida(bruto, parte, info, cfg, True, trecho)).returncode == 0
            if not ok:
                subprocess.run(cmd_saida(bruto, parte, info, cfg, False, trecho), check=True)
            lista.append(f"file '{parte.as_posix()}'")
            progresso(f"  rally {r.rally}/{len(rallies)} codificado")
        (tmp / "lista.txt").write_text("\n".join(lista), encoding="utf-8")
        subprocess.run(["ffmpeg", "-nostdin", "-loglevel", "error", "-y", "-f", "concat", "-safe", "0",
                        "-i", str(tmp / "lista.txt"), "-c", "copy", str(destino)], check=True)
    return destino
