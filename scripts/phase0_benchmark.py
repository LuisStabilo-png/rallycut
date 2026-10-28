"""Fase 0: baseline de desempenho (proxy, áudio e codificação de saída).

Mede, num vídeo de teste, o tempo de:
  1. gerar o proxy de análise (baixa resolução/fps) decodificando em NVDEC;
  2. extrair o áudio mono;
  3. codificar a saída (altura/codec da config) com NVENC.

Também amostra a VRAM usada via nvidia-smi e extrapola os tempos para um jogo
de 2 h. Tudo vai para `outputs/phase0/` (ignorado pelo git).

Uso:
    uv run python scripts/phase0_benchmark.py --video data/Teste1.mp4 --modalidade quadra
"""

import argparse
import json
import subprocess
import threading
import time
from pathlib import Path

from rallycut.config import carregar_config

DURACAO_JOGO_REF_S = 2 * 3600


def sondar_video(caminho: Path) -> dict:
    """Lê duração, resolução, fps e codec do primeiro stream de vídeo."""
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


class AmostradorVram:
    """Amostra a memória de GPU usada (MiB) em segundo plano."""

    def __init__(self, intervalo_s: float = 0.5):
        self.intervalo_s = intervalo_s
        self.amostras: list[int] = []
        self._parar = threading.Event()
        self._thread = threading.Thread(target=self._rodar, daemon=True)

    def _ler(self) -> int | None:
        try:
            saida = subprocess.run(
                ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                capture_output=True, text=True, check=True,
            ).stdout
            return int(saida.strip().splitlines()[0])
        except (subprocess.CalledProcessError, FileNotFoundError, ValueError):
            return None

    def _rodar(self):
        while not self._parar.is_set():
            valor = self._ler()
            if valor is not None:
                self.amostras.append(valor)
            self._parar.wait(self.intervalo_s)

    def __enter__(self):
        base = self._ler()
        self.base_mib = base if base is not None else 0
        self._thread.start()
        return self

    def __exit__(self, *_):
        self._parar.set()
        self._thread.join()

    @property
    def pico_extra_mib(self) -> int:
        return max(self.amostras, default=self.base_mib) - self.base_mib


def executar_etapa(nome: str, cmd: list[str], duracao_video_s: float) -> dict:
    """Roda um comando ffmpeg medindo tempo e VRAM; levanta erro com o stderr."""
    print(f"[{nome}] rodando...")
    with AmostradorVram() as vram:
        inicio = time.perf_counter()
        proc = subprocess.run(cmd, capture_output=True, text=True)
        tempo = time.perf_counter() - inicio
    if proc.returncode != 0:
        raise RuntimeError(f"{nome} falhou (código {proc.returncode}):\n{proc.stderr[-2000:]}")
    resultado = {
        "tempo_s": round(tempo, 2),
        "velocidade_x_tempo_real": round(duracao_video_s / tempo, 2),
        "extrapolado_jogo_2h_min": round(DURACAO_JOGO_REF_S / (duracao_video_s / tempo) / 60, 1),
        "vram_pico_extra_mib": vram.pico_extra_mib,
    }
    print(f"[{nome}] {tempo:.1f}s ({resultado['velocidade_x_tempo_real']}x tempo real, "
          f"~{resultado['extrapolado_jogo_2h_min']} min para 2 h, VRAM +{vram.pico_extra_mib} MiB)")
    return resultado


def largura_par(info: dict, altura: int) -> int:
    """Largura proporcional à altura desejada, arredondada para número par."""
    return round(info["largura"] * altura / info["altura"] / 2) * 2


def cmd_proxy(video: Path, destino: Path, info: dict, cfg: dict, nvdec: bool) -> list[str]:
    p = cfg["proxy"]
    largura = largura_par(info, p["altura"])
    if nvdec:
        pre = ["-hwaccel", "cuda", "-hwaccel_output_format", "cuda"]
        filtro = f"fps={p['fps']},scale_cuda={largura}:{p['altura']},hwdownload,format=nv12"
    else:
        pre = []
        filtro = f"fps={p['fps']},scale={largura}:{p['altura']}"
    return ["ffmpeg", "-nostdin", "-loglevel", "error", "-y", *pre, "-i", str(video),
            "-an", "-vf", filtro, "-c:v", "libx264", "-preset", "veryfast", "-crf", "28",
            str(destino)]


def cmd_audio(video: Path, destino: Path, cfg: dict) -> list[str]:
    return ["ffmpeg", "-nostdin", "-loglevel", "error", "-y", "-i", str(video), "-vn",
            "-ac", "1", "-ar", str(cfg["audio"]["taxa_hz"]), "-c:a", "pcm_s16le", str(destino)]


def cmd_saida(video: Path, destino: Path, info: dict, cfg: dict, nvdec: bool) -> list[str]:
    s = cfg["saida"]
    largura = largura_par(info, s["altura"])
    if nvdec:
        pre = ["-hwaccel", "cuda", "-hwaccel_output_format", "cuda"]
        filtro = f"scale_cuda={largura}:{s['altura']}"
    else:
        pre = []
        filtro = f"scale={largura}:{s['altura']}"
    fps = ["-r", str(s["fps"])] if s.get("fps") else []
    return ["ffmpeg", "-nostdin", "-loglevel", "error", "-y", *pre, "-i", str(video),
            "-vf", filtro, *fps, "-c:v", s["codec"], "-preset", s["preset"],
            "-rc", "vbr", "-cq", str(s["cq"]), "-b:v", "0",
            "-c:a", s["audio_codec"], "-b:a", s["audio_bitrate"], str(destino)]


def principal():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--video", type=Path, required=True)
    ap.add_argument("--modalidade", choices=["quadra", "praia"], default="quadra")
    ap.add_argument("--config", type=Path, default=None, help="YAML alternativo")
    ap.add_argument("--saida", type=Path, default=Path("outputs/phase0"))
    ap.add_argument("--sem-nvdec", action="store_true", help="decodifica na CPU (comparação)")
    args = ap.parse_args()

    cfg = carregar_config(args.modalidade, args.config)
    info = sondar_video(args.video)
    args.saida.mkdir(parents=True, exist_ok=True)
    nvdec = cfg["proxy"]["usar_nvdec"] and not args.sem_nvdec
    print(f"Vídeo: {info['largura']}x{info['altura']} {info['codec']} {info['fps']:.2f}fps "
          f"{info['duracao_s']:.0f}s | NVDEC={'sim' if nvdec else 'não'}")

    dur = info["duracao_s"]
    resultado = {"video": str(args.video), "info": info, "nvdec": nvdec, "etapas": {}}
    resultado["etapas"]["proxy"] = executar_etapa(
        "proxy", cmd_proxy(args.video, args.saida / "proxy.mp4", info, cfg, nvdec), dur)
    resultado["etapas"]["audio"] = executar_etapa(
        "audio", cmd_audio(args.video, args.saida / "audio.wav", cfg), dur)
    resultado["etapas"]["saida"] = executar_etapa(
        "saida", cmd_saida(args.video, args.saida / "saida.mp4", info, cfg, nvdec), dur)

    total = sum(e["tempo_s"] for e in resultado["etapas"].values())
    resultado["total_s"] = round(total, 2)
    resultado["total_extrapolado_jogo_2h_min"] = round(DURACAO_JOGO_REF_S / (dur / total) / 60, 1)
    destino = args.saida / "baseline.json"
    destino.write_text(json.dumps(resultado, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nTotal (sem a análise): {total:.1f}s -> ~{resultado['total_extrapolado_jogo_2h_min']} min "
          f"para um jogo de 2 h. Relatório: {destino}")


if __name__ == "__main__":
    principal()
