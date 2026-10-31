"""Mede quanto o app de revisão ajuda, usando as previsões de jogo de fora (outputs/modelos/probs_logo.npz).

Por par: quantos inícios o detector perde (rally inteiro ausente x rally que começa tarde) e quantos
dos ausentes aparecem entre os N primeiros candidatos do caçador de lacunas, para vários pisos.
Também mostra o esforço: itens a conferir (rallies + lacunas suspeitas) contra os rallies do gabarito.
Com --corrigir, simula um revisor perfeito (usa o gabarito) e mede o início preservado depois.
Com --exportar DIR --par PAR, grava o resultado dessa revisão como dado de treino em DIR/<PAR>_sim.

Uso:
    uv run python scripts/revisao_simulada.py [--pisos 0.1 0.15 0.25] [--corrigir]
"""

import argparse
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from rallycut import revisao as rv
from rallycut.config import carregar_config
from rallycut.deteccao import avaliar
from rallycut.features import JANELA_S, rotular_janelas
from rallycut.inferencia import rallies
from rallycut.pares import bruto_do_par


def principal():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pisos", type=float, nargs="+", default=[0.1, 0.15, 0.25])
    ap.add_argument("--corrigir", action="store_true")
    ap.add_argument("--exportar", type=Path, default=None)
    ap.add_argument("--par", default=None)
    args = ap.parse_args()

    probs = dict(np.load(Path("outputs/modelos/probs_logo.npz")))
    resumo, concordancia = [], {}
    for par, p in probs.items():
        mod, duracao = par.split("_")[0], len(p) * JANELA_S
        gab = pd.read_csv(Path("outputs/pares") / par / "trechos.csv")
        det = rallies(p, mod).assign(status="pendente")
        dentro = lambda t: ((det["ini_s"] <= t) & (t <= det["fim_s"])).any()  # noqa: E731
        ausentes = [r for r in gab.itertuples() if not any(det["ini_s"] < r.bruto_fim) or
                    not ((det["ini_s"] < r.bruto_fim) & (det["fim_s"] > r.bruto_ini)).any()]
        tardios = [r for r in gab.itertuples() if not dentro(r.bruto_ini) and r not in ausentes]
        for piso in args.pisos:
            lac = rv.lacunas(p, det, duracao, piso=piso, maximo=1000)
            ranks = []
            for r in ausentes:
                achou = lac.reset_index(drop=True)
                ok = achou[(achou["ini_s"] <= r.bruto_ini + 1) & (achou["fim_s"] >= r.bruto_ini)]
                ranks.append(int(ok.index[0]) + 1 if len(ok) else 10**6)
            resumo.append({"par": par, "piso": piso, "rallies": len(gab), "ausentes": len(ausentes), "tardios": len(tardios),
                           "top5": sum(k <= 5 for k in ranks), "top10": sum(k <= 10 for k in ranks),
                           "todas": sum(k < 10**6 for k in ranks), "candidatas": len(lac), "a_conferir": len(det) + len(lac)})
        if args.corrigir:
            novo = det.copy()
            for r in gab.itertuples():
                if not dentro(r.bruto_ini):  # revisor perfeito: corrige início tardio ou adiciona o rally inteiro
                    cob = novo[(novo["ini_s"] < r.bruto_fim) & (novo["fim_s"] > r.bruto_ini)]
                    margem = carregar_config(mod)["deteccao"]["margem_pre_s"]
                    ativos = rv.ativos(novo)
                    alvo = int(cob["rally"].iloc[0]) if len(cob) else None
                    antes_fim = ativos[(ativos["fim_s"] <= r.bruto_ini) & (ativos["rally"] != alvo)]["fim_s"].max()
                    ini = max(r.bruto_ini - margem, 0 if np.isnan(antes_fim) else antes_fim)
                    if len(cob):
                        novo = rv.ajustar(novo, alvo, ini, cob["fim_s"].iloc[0], duracao)
                    else:
                        depois_ini = ativos[ativos["ini_s"] >= r.bruto_fim]["ini_s"].min()
                        novo = rv.adicionar(novo, ini, min(r.bruto_fim + 1.5, duracao if np.isnan(depois_ini) else depois_ini), duracao)
            for k in novo.loc[novo["status"] == "pendente", "rally"]:
                novo = rv.aceitar(novo, int(k))
            d = carregar_config(mod)["deteccao"]
            t = np.arange(len(p)) * JANELA_S
            gab_rot = rotular_janelas(t, gab)
            for nome, prob in (("sem núcleo", None), ("com núcleo", p)):
                with tempfile.TemporaryDirectory() as tmp:
                    rv.exportar_treino(novo, Path(tmp) / "x", Path("v.mp4"), duracao, d["margem_pre_s"], d["margem_pos_s"], p=prob)
                    rot = rotular_janelas(t, pd.read_csv(Path(tmp) / "x" / "trechos.csv"))
                concordancia.setdefault(nome, []).append((rot == gab_rot).mean())
            antes = avaliar(gab, det[["ini_s", "fim_s"]].to_numpy())
            depois = avaliar(gab, rv.ativos(novo)[["ini_s", "fim_s"]].to_numpy())
            if args.exportar and par == args.par:
                for k in novo.loc[novo["status"] == "pendente", "rally"]:
                    novo = rv.aceitar(novo, int(k))
                d = carregar_config(mod)["deteccao"]
                rv.exportar_treino(novo, args.exportar / f"{par}_sim", bruto_do_par(Path("data/pares") / par), duracao,
                                   d["margem_pre_s"], d["margem_pos_s"])
            print(f"{par}: início preservado {antes['inicios_preservados']:.0%} -> {depois['inicios_preservados']:.0%} | "
                  f"saída {antes['mantido_min']:.1f} -> {depois['mantido_min']:.1f} min (gabarito {gab['duracao_s'].sum() / 60:.1f})")
    for nome, v in concordancia.items():
        print(f"rótulos exportados da revisão {nome}: concordam com o gabarito do editado em {np.mean(v):.0%} das janelas (por par: "
              + ", ".join(f"{x:.0%}" for x in v) + ")")
    r = pd.DataFrame(resumo)
    for piso, g in r.groupby("piso"):
        print(f"\n== piso {piso} ==\n{g.drop(columns='piso').to_string(index=False)}")
        a = g["ausentes"].sum()
        print(f"ausentes {a} | achados nos 5 primeiros {g['top5'].sum()} ({g['top5'].sum() / max(a, 1):.0%}) | nos 10 primeiros "
              f"{g['top10'].sum()} ({g['top10'].sum() / max(a, 1):.0%}) | em qualquer posição {g['todas'].sum()} | tardios {g['tardios'].sum()} | "
              f"a conferir {g['a_conferir'].sum()} de {g['rallies'].sum()} rallies do gabarito")


if __name__ == "__main__":
    principal()
