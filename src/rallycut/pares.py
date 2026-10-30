"""Pares bruto/editado em data/pares/<modalidade>_<nome>/: prepara o dataset de treino.

Para cada par, em outputs/pares/<par>/ (só refaz o que falta):
  trechos.csv ...  alinhamento bruto <-> editado (o gabarito)
  janelas_<altura>.csv ...  features por janela de 0,5 s + `rotulo` (1 = rally, 0 = tempo morto)
  embeddings_<altura>.npy . embedding visual (960) de cada janela
(<altura> = resolução da detecção de jogadores; os arquivos de 360p não têm sufixo)
"""

from pathlib import Path

import numpy as np
import pandas as pd

from rallycut.alinhamento import alinhar_par
from rallycut.config import MODALIDADES, carregar_config
from rallycut.features import carregar_embedder, features_video, rotular_janelas, sufixo
from rallycut.media import sondar_video
from rallycut.pessoas import carregar_detector


def listar_pares(raiz: Path = Path("data/pares")) -> list[Path]:
    return sorted(p for p in raiz.iterdir() if p.is_dir()
                  and list(p.glob("*bruto*")) and list(p.glob("*editado*")))


def preparar_par(pasta: Path, saida_base: Path = Path("outputs/pares"), progresso=print,
                 altura: int | None = None) -> Path:
    """Alinha e extrai as features do par (se ainda não existirem). Devolve a pasta de saída.
    `altura`: resolução da detecção de jogadores (padrão: config `features.altura`)."""
    modalidade = pasta.name.split("_")[0]
    if modalidade not in MODALIDADES:
        raise ValueError(f"nome da pasta deve começar com quadra_ ou praia_: {pasta.name}")
    bruto, editado = next(pasta.glob("*bruto*")), next(pasta.glob("*editado*"))
    saida, cfg = saida_base / pasta.name, carregar_config(modalidade)
    if not (saida / "trechos.csv").exists():
        progresso(f"[{pasta.name}] alinhando bruto e editado...")
        alinhar_par(bruto, editado, saida, cfg)
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
