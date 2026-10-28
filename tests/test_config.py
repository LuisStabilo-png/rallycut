import pytest

from rallycut.config import carregar_config


def test_praia_sobrescreve_peso_audio():
    cfg = carregar_config("praia")
    assert cfg["peso_audio"] == 0.5
    assert cfg["proxy"]["fps"] == 8  # herdado de "geral"


def test_quadra_herda_geral():
    cfg = carregar_config("quadra")
    assert cfg["saida"]["codec"] == "h264_nvenc"
    assert cfg["peso_audio"] == 1.0


def test_modalidade_invalida():
    with pytest.raises(ValueError):
        carregar_config("areia")
