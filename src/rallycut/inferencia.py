"""Do vídeo bruto (sem gabarito) às probabilidades de rally e aos rallies: o que o `cortar.py` usa."""

import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

from rallycut.bola import carregar_mapa, extrair_candidatos, features_bola
from rallycut.config import carregar_config
from rallycut.deteccao import segmentar
from rallycut.features import JANELA_S, carregar_embedder, features_video, sufixo
from rallycut.media import cmd_audio, sondar_video
from rallycut.pessoas import carregar_detector
from rallycut.temporal import carregar_tcn, prever_video


def tabela_de_features(bruto: Path, saida: Path, modalidade: str, mapa: Path, pca: dict, colunas: list[str],
                       progresso=print) -> np.ndarray:
    """Features por janela de 0,5 s do vídeo, na ordem de `colunas` (as do treino do modelo).

    Tudo que é caro fica em cache em `saida` (áudio, janelas, embeddings e candidatos de bola)."""
    cfg = carregar_config(modalidade)
    saida.mkdir(parents=True, exist_ok=True)
    altura = cfg["features"]["altura"]
    suf = sufixo(altura)
    duracao = sondar_video(bruto)["duracao_s"]
    wav = saida / "audio_bruto.wav"
    if not wav.exists():
        progresso("extraindo o áudio...")
        subprocess.run(cmd_audio(bruto, wav, cfg), check=True)
    if not (saida / f"janelas{suf}.csv").exists() or not (saida / f"embeddings{suf}.npy").exists():
        progresso(f"extraindo features ({altura}p)...")
        f, emb = features_video(bruto, wav, duracao, carregar_detector(), carregar_embedder(), altura)
        np.save(saida / f"embeddings{suf}.npy", emb)
        f.round(5).to_csv(saida / f"janelas{suf}.csv", index=False)
    d = pd.read_csv(saida / f"janelas{suf}.csv")
    emb = np.load(saida / f"embeddings{suf}.npy").astype(np.float32)
    proj = (emb - pca["media"]) @ pca["componentes"].T
    d[[f"emb{i}" for i in range(proj.shape[1])]] = proj
    d["eh_praia"] = int(modalidade == "praia")
    if not (saida / "bola_candidatos.csv").exists():
        progresso("detectando a bola...")
        extrair_candidatos(bruto, carregar_mapa(mapa)).round(2).to_csv(saida / "bola_candidatos.csv", index=False)
    bola = features_bola(pd.read_csv(saida / "bola_candidatos.csv"), len(d))
    return pd.concat([d, bola], axis=1).reindex(columns=colunas).to_numpy(float)


def probabilidades(bruto: Path, saida: Path, modalidade: str, modelo: Path, mapa: Path, progresso=print) -> np.ndarray:
    """Probabilidade de rally por janela de 0,5 s (em cache: `probs.npy`)."""
    destino = saida / "probs.npy"
    if destino.exists():
        return np.load(destino)
    modelos, colunas, pca = carregar_tcn(modelo)
    p = prever_video(modelos, tabela_de_features(bruto, saida, modalidade, mapa, pca, colunas, progresso))
    np.save(destino, p)
    return p


def rallies(p: np.ndarray, modalidade: str) -> pd.DataFrame:
    """Rallies (início e fim em s do bruto) a partir das probabilidades, com o limiar e as margens da config.
    `confianca` = probabilidade média das janelas acima do limiar dentro do rally (baixa = revisar primeiro)."""
    cfg = carregar_config(modalidade)["deteccao"]
    segs = segmentar(p, cfg, len(p) * JANELA_S)
    conf = []
    for a, b in segs:
        dentro = p[int(a / JANELA_S):int(np.ceil(b / JANELA_S))]
        acima = dentro[dentro >= cfg["limiar"]]
        conf.append(float(acima.mean()) if len(acima) else 0.0)
    return pd.DataFrame({"rally": np.arange(1, len(segs) + 1), "ini_s": segs[:, 0].round(3),
                         "fim_s": segs[:, 1].round(3), "duracao_s": (segs[:, 1] - segs[:, 0]).round(3),
                         "confianca": np.round(conf, 3)})
