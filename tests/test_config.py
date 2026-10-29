import pytest

from rallycut.config import carregar_config


def test_modalidade_herda_geral():
    for m in ("quadra", "praia"):
        cfg = carregar_config(m)
        assert cfg["proxy"]["fps"] == 8
        assert cfg["deteccao"]["limiar"] == 0.4


def test_modalidade_sobrescreve_geral(tmp_path):
    yaml = tmp_path / "c.yaml"
    yaml.write_text("geral:\n  deteccao:\n    limiar: 0.4\n    margem_pre_s: 1.5\npraia:\n  deteccao:\n    limiar: 0.3\n")
    praia = carregar_config("praia", yaml)["deteccao"]
    assert praia == {"limiar": 0.3, "margem_pre_s": 1.5}  # mescla recursiva
    assert carregar_config("quadra", yaml)["deteccao"]["limiar"] == 0.4


def test_modalidade_invalida():
    with pytest.raises(ValueError):
        carregar_config("areia")
