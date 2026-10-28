"""Utilitários de mídia: sondagem com ffprobe e montagem de comandos ffmpeg."""

import json
import subprocess
from pathlib import Path


def sondar_video(caminho: Path | str) -> dict:
    """Lê codec, resolução, fps, pix_fmt e duração do primeiro stream de vídeo."""
    saida = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=codec_name,width,height,r_frame_rate,pix_fmt:format=duration",
         "-of", "json", str(caminho)],
        capture_output=True, text=True, check=True,
    ).stdout
    dados = json.loads(saida)
    stream = dados["streams"][0]
    num, den = stream["r_frame_rate"].split("/")
    return {
        "codec": stream["codec_name"],
        "largura": stream["width"],
        "altura": stream["height"],
        "fps": float(num) / float(den),
        "pix_fmt": stream["pix_fmt"],
        "duracao_s": float(dados["format"]["duration"]),
    }


def largura_par(info: dict, altura: int) -> int:
    """Largura proporcional à altura desejada, arredondada para número par."""
    return round(info["largura"] * altura / info["altura"] / 2) * 2


def _base(video: Path, nvdec: bool) -> list[str]:
    pre = ["-hwaccel", "cuda", "-hwaccel_output_format", "cuda"] if nvdec else []
    return ["ffmpeg", "-nostdin", "-loglevel", "error", "-y", *pre, "-i", str(video)]


def cmd_proxy(video: Path, destino: Path, info: dict, cfg: dict, nvdec: bool) -> list[str]:
    """Proxy de análise: baixa resolução e baixo fps, sem áudio."""
    p = cfg["proxy"]
    largura = largura_par(info, p["altura"])
    if nvdec:
        filtro = f"fps={p['fps']},scale_cuda={largura}:{p['altura']},hwdownload,format=nv12"
    else:
        filtro = f"fps={p['fps']},scale={largura}:{p['altura']}"
    return [*_base(video, nvdec), "-an", "-vf", filtro,
            "-c:v", "libx264", "-preset", "veryfast", "-crf", "28", str(destino)]


def cmd_audio(video: Path, destino: Path, cfg: dict) -> list[str]:
    """Áudio mono em WAV PCM 16 bits na taxa configurada."""
    return [*_base(video, False), "-vn", "-ac", "1", "-ar", str(cfg["audio"]["taxa_hz"]),
            "-c:a", "pcm_s16le", str(destino)]


def cmd_saida(video: Path, destino: Path, info: dict, cfg: dict, nvdec: bool) -> list[str]:
    """Codificação final (altura, codec e qualidade da config)."""
    s = cfg["saida"]
    largura = largura_par(info, s["altura"])
    filtro = f"scale_cuda={largura}:{s['altura']}" if nvdec else f"scale={largura}:{s['altura']}"
    fps = ["-r", str(s["fps"])] if s.get("fps") else []
    return [*_base(video, nvdec), "-vf", filtro, *fps,
            "-c:v", s["codec"], "-preset", s["preset"], "-rc", "vbr", "-cq", str(s["cq"]),
            "-b:v", "0", "-c:a", s["audio_codec"], "-b:a", s["audio_bitrate"], str(destino)]
