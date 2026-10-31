"""Pares em data/pares/<modalidade>_<nome>/ (bruto + editado) e data/revisados/<modalidade>_<nome>/
(trechos.csv + bruto.txt com o caminho do vídeo, exportados pelo app de revisão): prepara o dataset de treino.

Para cada par, em outputs/pares/<par>/ (só refaz o que falta):
  trechos.csv ...  alinhamento bruto <-> editado (o gabarito)
  janelas_<altura>.csv ...  features por janela de 0,5 s + `rotulo` (1 = rally, 0 = tempo morto)
  embeddings_<altura>.npy . embedding visual (960) de cada janela
(<altura> = resolução da detecção de jogadores; os arquivos de 360p não têm sufixo)
"""

import shutil
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

from rallycut.alinhamento import alinhar_par
from rallycut.config import MODALIDADES, carregar_config
from rallycut.features import carregar_embedder, features_video, rotular_janelas, sufixo
from rallycut.media import cmd_audio, sondar_video
from rallycut.pessoas import carregar_detector


RAIZES = (Path("data/pares"), Path("data/revisados"))


def bruto_do_par(pasta: Path) -> Path | None:
    """Vídeo bruto do par: o arquivo `*bruto*` ou, nos revisados, o caminho escrito em `bruto.txt`."""
    if (pasta / "bruto.txt").exists():
        return Path((pasta / "bruto.txt").read_text(encoding="utf-8").strip())
    return next(pasta.glob("*bruto*"), None)


def listar_pares(raiz: Path | None = None) -> list[Path]:
    """Pares com gabarito (bruto + editado, ou trechos.csv + bruto.txt dos revisados) em `raiz` (padrão:
    data/pares e data/revisados). Pastas com um arquivo `IGNORAR` (gabarito não confiável) ficam de fora."""
    def valido(p: Path) -> bool:
        gabarito = (p / "trechos.csv").exists() if (p / "bruto.txt").exists() else bool(list(p.glob("*editado*")))
        return p.is_dir() and not (p / "IGNORAR").exists() and bruto_do_par(p) is not None and gabarito

    return sorted(p for r in ([raiz] if raiz else RAIZES) if r.is_dir() for p in r.iterdir() if valido(p))


def preparar_par(pasta: Path, saida_base: Path = Path("outputs/pares"), progresso=print,
                 altura: int | None = None) -> Path:
    """Alinha e extrai as features do par (se ainda não existirem). Devolve a pasta de saída.
    `altura`: resolução da detecção de jogadores (padrão: config `features.altura`)."""
    modalidade = pasta.name.split("_")[0]
    if modalidade not in MODALIDADES:
        raise ValueError(f"nome da pasta deve começar com quadra_ ou praia_: {pasta.name}")
    bruto = bruto_do_par(pasta)
    saida, cfg = saida_base / pasta.name, carregar_config(modalidade)
    if (pasta / "bruto.txt").exists():  # revisado: o gabarito já vem pronto, só falta o áudio
        saida.mkdir(parents=True, exist_ok=True)
        for arq in ("trechos.csv", "removidos.csv"):
            if not (saida / arq).exists():
                shutil.copy(pasta / arq, saida / arq)
        if not (saida / "audio_bruto.wav").exists():
            subprocess.run(cmd_audio(bruto, saida / "audio_bruto.wav", cfg), check=True)
    elif not (saida / "trechos.csv").exists():
        progresso(f"[{pasta.name}] alinhando bruto e editado...")
        alinhar_par(bruto, next(pasta.glob("*editado*")), saida, cfg)
    altura = cfg["features"]["altura"] if altura is None else altura
    suf = sufixo(altura)
    if not (saida / f"janelas{suf}.csv").exists() or not (saida / f"embeddings{suf}.npy").exists():
        progresso(f"[{pasta.name}] extraindo features ({altura}p)...")
        info = sondar_video(bruto)
        f, emb = features_video(bruto, saida / "audio_bruto.wav", info["duracao_s"],
                                carregar_detector(), carregar_embedder(), altura)
        np.save(saida / f"embeddings{suf}.npy", emb)
        f["rotulo"] = rotular_janelas(f["t"].to_numpy(), pd.read_csv(saida / "trechos.csv"))
        f.round(5).to_csv(saida / f"janelas{suf}.csv", index=False)
    return saida
