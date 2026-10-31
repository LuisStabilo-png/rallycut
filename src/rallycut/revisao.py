"""Lógica da revisão manual dos rallies (sem interface): estados, validação, lacunas e exportação.

Um `rallies.csv` do cortar.py vira uma tabela com `status`: pendente (ainda não olhado), aceito,
ajustado, removido ou adicionado. A revisão fica em `rallies_revisados.csv`, na mesma pasta."""

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from rallycut.deteccao import segmentar
from rallycut.features import JANELA_S

COLUNAS = ["rally", "ini_s", "fim_s", "duracao_s", "confianca", "status"]


def abrir(pasta: Path) -> pd.DataFrame:
    """Revisão em andamento, ou os rallies detectados (todos pendentes)."""
    revisado = pasta / "rallies_revisados.csv"
    if revisado.exists():
        return pd.read_csv(revisado)
    df = pd.read_csv(pasta / "rallies.csv")
    return df.assign(status="pendente")[COLUNAS]


def salvar(df: pd.DataFrame, pasta: Path):
    """Grava `rallies_revisados.csv` sem arriscar um arquivo pela metade."""
    tmp = pasta / "rallies_revisados.csv.tmp"
    df.to_csv(tmp, index=False)
    os.replace(tmp, pasta / "rallies_revisados.csv")


def ativos(df: pd.DataFrame) -> pd.DataFrame:
    return df[df["status"] != "removido"].sort_values("ini_s")


def fila(df: pd.DataFrame) -> pd.DataFrame:
    """Ordem de revisão: pendentes do menos ao mais confiante, depois o resto pelo número."""
    pend = df[df["status"] == "pendente"].sort_values(["confianca", "rally"])
    return pd.concat([pend, df[df["status"] != "pendente"].sort_values("rally")])


def _validar(df: pd.DataFrame, ini: float, fim: float, duracao_video: float, ignorar: int | None, minimo_s: float):
    if not (0 <= ini < fim <= duracao_video):
        raise ValueError(f"início e fim devem satisfazer 0 <= início < fim <= {duracao_video:.1f} s")
    if fim - ini < minimo_s:
        raise ValueError(f"o rally deve ter pelo menos {minimo_s:g} s")
    for r in ativos(df).itertuples():
        if r.rally != ignorar and ini < r.fim_s and r.ini_s < fim:
            raise ValueError(f"sobrepõe o rally {r.rally} ({r.ini_s:.1f}-{r.fim_s:.1f} s)")


def _linha(df: pd.DataFrame, k: int) -> int:
    return int(df.index[df["rally"] == k][0])


def aceitar(df: pd.DataFrame, k: int) -> pd.DataFrame:
    df = df.copy()
    i = _linha(df, k)
    if df.loc[i, "status"] in ("pendente", "removido"):
        df.loc[i, "status"] = "aceito"
    return df


def ajustar(df: pd.DataFrame, k: int, ini: float, fim: float, duracao_video: float, minimo_s: float = 1.0) -> pd.DataFrame:
    _validar(df, ini, fim, duracao_video, k, minimo_s)
    df = df.copy()
    i = _linha(df, k)
    df.loc[i, ["ini_s", "fim_s", "duracao_s"]] = [round(ini, 3), round(fim, 3), round(fim - ini, 3)]
    if df.loc[i, "status"] != "adicionado":
        df.loc[i, "status"] = "ajustado"
    return df


def remover(df: pd.DataFrame, k: int) -> pd.DataFrame:
    df = df.copy()
    df.loc[_linha(df, k), "status"] = "removido"
    return df


def adicionar(df: pd.DataFrame, ini: float, fim: float, duracao_video: float, minimo_s: float = 1.0) -> pd.DataFrame:
    _validar(df, ini, fim, duracao_video, None, minimo_s)
    nova = {"rally": int(df["rally"].max()) + 1 if len(df) else 1, "ini_s": round(ini, 3), "fim_s": round(fim, 3),
            "duracao_s": round(fim - ini, 3), "confianca": np.nan, "status": "adicionado"}
    return pd.concat([df, pd.DataFrame([nova])], ignore_index=True)


def lacunas(p: np.ndarray, df: pd.DataFrame, duracao_video: float, piso: float = 0.15, minimo_s: float = 3.0,
            maximo: int = 30) -> pd.DataFrame:
    """Trechos fora dos rallies ativos onde o modelo ainda vê atividade: candidatos a rally perdido.

    `pico` = maior média de 3 s da probabilidade dentro do vazio; `t_pico` = onde ocorre. Do mais ao menos provável."""
    a = ativos(df)
    if len(a):
        bordas = [(0.0, a["ini_s"].iloc[0]), *zip(a["fim_s"].iloc[:-1], a["ini_s"].iloc[1:]), (a["fim_s"].iloc[-1], duracao_video)]
    else:
        bordas = [(0.0, duracao_video)]
    linhas = []
    for ini, fim in bordas:
        if fim - ini < minimo_s:
            continue
        i0, i1 = int(ini / JANELA_S), int(np.ceil(fim / JANELA_S))
        suave = pd.Series(p[i0:i1]).rolling(6, center=True).mean()  # só janelas inteiras de 3 s dentro do vazio
        if suave.notna().any() and suave.max() >= piso:
            linhas.append({"ini_s": round(ini, 1), "fim_s": round(fim, 1), "pico": round(float(suave.max()), 3),
                           "t_pico": round((i0 + int(suave.idxmax())) * JANELA_S, 1)})
    return pd.DataFrame(linhas, columns=["ini_s", "fim_s", "pico", "t_pico"]).sort_values("pico", ascending=False).head(maximo)


def marcas_lacunas(pasta: Path) -> list[float]:
    arq = pasta / "lacunas_revisadas.json"
    return json.loads(arq.read_text()) if arq.exists() else []


def marcar_lacuna(pasta: Path, t_pico: float):
    marcas = marcas_lacunas(pasta)
    if t_pico not in marcas:
        (pasta / "lacunas_revisadas.json").write_text(json.dumps(marcas + [t_pico]))


def lacuna_revisada(t_pico: float, marcas: list[float], tolerancia_s: float = 5.0) -> bool:
    return any(abs(t_pico - m) <= tolerancia_s for m in marcas)


def pendencias(df: pd.DataFrame, lac: pd.DataFrame, marcas: list[float]) -> tuple[int, int]:
    """(rallies ainda pendentes, lacunas suspeitas ainda não marcadas)."""
    return int((df["status"] == "pendente").sum()), int(sum(not lacuna_revisada(t, marcas) for t in lac["t_pico"]))


def _nucleos(p: np.ndarray, ini: float, fim: float, limiar: float, duracao_video: float) -> list[tuple[float, float]]:
    """Partes de [ini, fim] onde o modelo tem muita certeza de rally (limiar alto, vazios curtos unidos)."""
    cfg = {"limiar": limiar, "suavizar_s": 3.0, "uniao_gap_s": 3.0, "duracao_min_s": 1.0, "margem_pre_s": 0.0, "margem_pos_s": 0.0}
    return [(max(a, ini), min(b, fim)) for a, b in segmentar(p, cfg, duracao_video) if a < fim and b > ini]


def exportar_treino(df: pd.DataFrame, destino: Path, bruto: Path, duracao_video: float,
                    margem_pre_s: float, margem_pos_s: float, p: np.ndarray | None = None, limiar_nucleo: float = 0.7):
    """Grava `trechos.csv`, `removidos.csv` e `bruto.txt` (formato do gabarito) em `destino`.

    Ajustados e adicionados ficam como você os deixou. Um rally aceito sem mudança só foi confirmado como
    rally: o vídeo cortado traz tempo morto junto (as margens e o limiar baixo do detector), então, com as
    probabilidades `p`, ele vira o núcleo onde o modelo tem muita certeza (limiar `limiar_nucleo`); sem `p`,
    perde só as margens."""
    linhas, acum = [], 0.0
    trechos = []
    for r in ativos(df).itertuples():
        if r.status != "aceito":
            trechos.append((r.ini_s, r.fim_s))
        elif p is not None and (nucleos := _nucleos(p, r.ini_s, r.fim_s, limiar_nucleo, duracao_video)):
            trechos += nucleos
        else:
            trechos.append((r.ini_s + margem_pre_s, r.fim_s - margem_pos_s))
    for ini, fim in trechos:
        if fim - ini <= 0:
            continue
        linhas.append({"rally": len(linhas) + 1, "editado_ini": round(acum, 3), "editado_fim": round(acum + fim - ini, 3),
                       "bruto_ini": round(ini, 3), "bruto_fim": round(fim, 3), "custo": 0.0, "ajuste_s": 0.0, "pico": 0.0,
                       "duracao_s": round(fim - ini, 3)})
        acum += fim - ini
    t = pd.DataFrame(linhas)
    fora, anterior = [], 0.0
    for r in t.itertuples():
        if r.bruto_ini > anterior:
            fora.append({"bruto_ini": anterior, "bruto_fim": r.bruto_ini, "duracao_s": round(r.bruto_ini - anterior, 3)})
        anterior = r.bruto_fim
    if duracao_video > anterior:
        fora.append({"bruto_ini": anterior, "bruto_fim": duracao_video, "duracao_s": round(duracao_video - anterior, 3)})
    destino.mkdir(parents=True, exist_ok=True)
    t.to_csv(destino / "trechos.csv", index=False)
    pd.DataFrame(fora, columns=["bruto_ini", "bruto_fim", "duracao_s"]).to_csv(destino / "removidos.csv", index=False)
    (destino / "bruto.txt").write_text(str(bruto.resolve()), encoding="utf-8")
