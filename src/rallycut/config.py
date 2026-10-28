"""Carregamento da configuração (YAML) com mesclagem por modalidade."""

from pathlib import Path

import yaml

MODALIDADES = ("quadra", "praia")
CONFIG_PADRAO = Path(__file__).resolve().parents[2] / "configs" / "default.yaml"


def _mesclar(base: dict, extra: dict) -> dict:
    """Mescla `extra` sobre `base` recursivamente, sem alterar os originais."""
    saida = dict(base)
    for chave, valor in extra.items():
        if isinstance(valor, dict) and isinstance(saida.get(chave), dict):
            saida[chave] = _mesclar(saida[chave], valor)
        else:
            saida[chave] = valor
    return saida


def carregar_config(modalidade: str, caminho: Path | str | None = None) -> dict:
    """Devolve a configuração efetiva: seção "geral" + seção da modalidade."""
    if modalidade not in MODALIDADES:
        raise ValueError(f"modalidade inválida: {modalidade!r} (use {MODALIDADES})")
    with open(caminho or CONFIG_PADRAO, encoding="utf-8") as f:
        bruto = yaml.safe_load(f)
    return _mesclar(bruto.get("geral", {}), bruto.get(modalidade, {}))
