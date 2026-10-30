import numpy as np
from pathlib import Path

from rallycut.config import carregar_config
from rallycut.features import JANELA_S
from rallycut.inferencia import rallies
from rallycut.media import cmd_saida

INFO = {"largura": 3840, "altura": 2160}


def test_rallies_aplica_limiar_e_margens_da_config():
    p = np.zeros(int(200 / JANELA_S))
    p[int(50 / JANELA_S):int(60 / JANELA_S)] = 0.9
    r = rallies(p, "quadra")
    margem_pre = carregar_config("quadra")["deteccao"]["margem_pre_s"]
    assert list(r.columns) == ["rally", "ini_s", "fim_s", "duracao_s", "confianca"] and len(r) == 1
    assert r["ini_s"][0] == 50 - margem_pre and r["confianca"][0] == 0.9


def test_cmd_saida_com_trecho_busca_por_tempo_antes_da_entrada():
    cfg = carregar_config("praia")
    cmd = cmd_saida(Path("v.mp4"), Path("p.mp4"), INFO, cfg, True, (12.5, 20.0))
    assert cmd[cmd.index("-ss") + 1] == "12.500" and cmd[cmd.index("-t") + 1] == "7.500"
    assert cmd.index("-ss") < cmd.index("-i") and "-ss" not in cmd_saida(Path("v.mp4"), Path("p.mp4"), INFO, cfg, True)
