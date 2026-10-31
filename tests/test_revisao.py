import numpy as np
import pandas as pd
import pytest

from rallycut import revisao as rv
from rallycut.features import JANELA_S

DUR = 200.0


def _df():
    return pd.DataFrame({"rally": [1, 2, 3], "ini_s": [10.0, 50.0, 90.0], "fim_s": [30.0, 70.0, 110.0],
                         "duracao_s": [20.0, 20.0, 20.0], "confianca": [0.9, 0.5, 0.7], "status": "pendente"})


def test_fila_comeca_pelos_menos_confiantes_e_deixa_revisados_para_depois():
    df = rv.aceitar(_df(), 2)
    assert list(rv.fila(df)["rally"]) == [3, 1, 2]


def test_ajustar_validar_e_sobreposicao():
    df = rv.ajustar(_df(), 1, 8.0, 35.0, DUR)
    assert df.loc[0, "status"] == "ajustado" and df.loc[0, "duracao_s"] == 27.0
    with pytest.raises(ValueError, match="sobrepõe o rally 2"):
        rv.ajustar(df, 1, 8.0, 55.0, DUR)
    with pytest.raises(ValueError):
        rv.ajustar(df, 1, 30.0, 20.0, DUR)
    with pytest.raises(ValueError):
        rv.adicionar(df, 190.0, 210.0, DUR)


def test_remover_libera_o_lugar_e_adicionar_cria_novo_rally():
    df = rv.remover(_df(), 2)
    df = rv.adicionar(df, 45.0, 75.0, DUR)
    assert list(rv.ativos(df)["rally"]) == [1, 4, 3] and df.loc[df["rally"] == 4, "status"].iloc[0] == "adicionado"


def test_lacunas_acham_atividade_fora_dos_rallies_e_marcas_as_dispensam(tmp_path):
    p = np.zeros(int(DUR / JANELA_S))
    p[int(140 / JANELA_S):int(150 / JANELA_S)] = 0.8
    p[int(31 / JANELA_S)] = 0.6  # pico de uma janela só: a média de 3 s o dilui abaixo do piso
    lac = rv.lacunas(p, _df(), DUR)
    assert len(lac) == 1 and 140 <= lac["t_pico"].iloc[0] <= 150
    assert rv.pendencias(_df(), lac, []) == (3, 1)
    rv.marcar_lacuna(tmp_path, float(lac["t_pico"].iloc[0]))
    tudo = rv.aceitar(rv.aceitar(rv.aceitar(_df(), 1), 2), 3)
    assert rv.pendencias(tudo, lac, rv.marcas_lacunas(tmp_path)) == (0, 0)


def test_salvar_e_abrir_voltam_igual(tmp_path):
    _df()[rv.COLUNAS].to_csv(tmp_path / "rallies.csv", index=False)
    df = rv.remover(rv.abrir(tmp_path), 3)
    rv.salvar(df, tmp_path)
    assert rv.abrir(tmp_path).equals(df) and not (tmp_path / "rallies_revisados.csv.tmp").exists()


def test_exportar_treino_tira_margens_dos_aceitos_e_mantem_os_ajustados(tmp_path):
    df = rv.aceitar(rv.ajustar(rv.aceitar(_df(), 1), 2, 48.0, 72.0, DUR), 3)
    rv.exportar_treino(df, tmp_path / "quadra_x", tmp_path / "v.mp4", DUR, margem_pre_s=4.0, margem_pos_s=1.5)
    t = pd.read_csv(tmp_path / "quadra_x" / "trechos.csv")
    assert list(t["bruto_ini"]) == [14.0, 48.0, 94.0] and list(t["bruto_fim"]) == [28.5, 72.0, 108.5]
    assert t["editado_fim"].iloc[-1] == pytest.approx(t["duracao_s"].sum())
    r = pd.read_csv(tmp_path / "quadra_x" / "removidos.csv")
    assert r["duracao_s"].sum() + t["duracao_s"].sum() == pytest.approx(DUR)
    assert (tmp_path / "quadra_x" / "bruto.txt").read_text().endswith("v.mp4")
