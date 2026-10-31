"""App de revisão dos cortes (Gradio, local).

Abre um resultado do cortar.py (outputs/cortes/<nome>/), mostra os rallies do menos ao mais confiante e os
vazios onde pode haver um rally perdido, e deixa corrigir tudo vendo o vídeo. Cada ação grava
`rallies_revisados.csv`. Dali dá para cortar o vídeo revisado ou exportar como dado de treino.

Uso:
    uv run python app/revisar.py
"""

import json
import subprocess
from pathlib import Path

import gradio as gr
import numpy as np
import pandas as pd

from rallycut import revisao as rv
from rallycut.config import carregar_config
from rallycut.corte import cortar_video
from rallycut.media import cmd_proxy_revisao, sondar_video

CORTES = Path("outputs/cortes")
COLS_RALLY = ["rally", "ini_s", "fim_s", "duracao_s", "confianca", "status"]
S: dict = {}  # estado da sessão (app local, um usuário)


def _pastas() -> list[str]:
    return sorted(p.name for p in CORTES.glob("*/") if (p / "rallies.csv").exists())


def _player(t0: float, t1: float) -> str:
    src = f"/gradio_api/file={S['proxy'].resolve().as_posix()}#t={max(t0, 0):.2f},{t1:.2f}"
    return f'<video src="{src}" controls style="width:100%;max-height:480px;background:#000"></video>'


def _contexto(ini: float, fim: float) -> str:
    r = S["cfg"]["revisao"]
    return _player(ini - r["contexto_antes_s"], fim + r["contexto_depois_s"])


def _lacunas() -> pd.DataFrame:
    r = S["cfg"]["revisao"]
    lac = rv.lacunas(S["p"], S["df"], S["duracao"], r["piso_lacuna"], maximo=r["max_lacunas"])
    marcas = rv.marcas_lacunas(S["pasta"])
    return lac.assign(revisada=["sim" if rv.lacuna_revisada(t, marcas) else "não" for t in lac["t_pico"]]).reset_index(drop=True)


def _atualizar() -> tuple:
    """Grava a revisão e devolve (tabela de rallies, tabela de lacunas, resumo)."""
    rv.salvar(S["df"], S["pasta"])
    fila = rv.fila(S["df"])
    S["ordem"], S["lac"] = list(fila["rally"]), _lacunas()
    pend, lac_pend = rv.pendencias(S["df"], S["lac"], rv.marcas_lacunas(S["pasta"]))
    n = S["df"]["status"].value_counts().to_dict()
    resumo = (f"**{pend}** rallies pendentes · **{lac_pend}** vazios suspeitos sem marca · "
              + " · ".join(f"{k}: {v}" for k, v in n.items() if k != "pendente")
              + (" · ✅ revisão completa (pode exportar como dado de treino)" if pend == lac_pend == 0 else ""))
    return fila[COLS_RALLY].round(2), S["lac"], resumo


def _selecionar(k: int):
    S["atual"] = k
    r = S["df"][S["df"]["rally"] == k].iloc[0]
    return _contexto(r.ini_s, r.fim_s), r.ini_s, r.fim_s, f"Rally {k} ({r.status}, confiança {r.confianca})"


def carregar(nome: str, bruto: str, modalidade: str, progresso=gr.Progress()):
    pasta = CORTES / nome
    fonte = json.loads((pasta / "fonte.json").read_text()) if (pasta / "fonte.json").exists() else {}
    bruto, modalidade = Path(bruto or fonte.get("video", "")), modalidade or fonte.get("modalidade")
    if not bruto.is_file():
        raise gr.Error("Informe o caminho do vídeo bruto.")
    cfg = carregar_config(modalidade)
    info = sondar_video(bruto)
    proxy = pasta / "proxy_revisao.mp4"
    if not proxy.exists():
        progresso(0.1, desc="Criando o proxy para assistir (uma vez, alguns minutos)...")
        if subprocess.run(cmd_proxy_revisao(bruto, proxy, info, cfg, True)).returncode:
            subprocess.run(cmd_proxy_revisao(bruto, proxy, info, cfg, False), check=True)
    S.update(pasta=pasta, bruto=bruto, mod=modalidade, cfg=cfg, duracao=info["duracao_s"], proxy=proxy,
             p=np.load(pasta / "probs.npy"), df=rv.abrir(pasta), atual=None)
    tabela, lac, resumo = _atualizar()
    return tabela, lac, resumo, f"{modalidade}_{bruto.stem.replace('_bruto', '')}"


def _acao(f, *args):
    """Aplica uma operação de `rv`; erros de validação (ou tempo inválido) viram aviso, sem mudar nada."""
    try:
        S["df"] = f(S["df"], *[float(a) if isinstance(a, str) else a for a in args])
    except (ValueError, TypeError) as e:
        gr.Warning(str(e))
    return _atualizar()


def aceitar_e_proximo():
    if S.get("atual") is not None:
        S["df"] = rv.aceitar(S["df"], S["atual"])
    tabela, lac, resumo = _atualizar()
    pend = [k for k in S["ordem"] if S["df"].loc[S["df"]["rally"] == k, "status"].iloc[0] == "pendente"]
    if not pend:
        return tabela, lac, resumo, gr.skip(), gr.skip(), gr.skip(), "Nenhum rally pendente."
    return (tabela, lac, resumo, *_selecionar(pend[0]))


def ajustar(ini: float, fim: float):
    r = S["cfg"]["revisao"]
    return _acao(rv.ajustar, S["atual"], ini, fim, S["duracao"], r["duracao_min_s"])


def remover():
    return _acao(rv.remover, S["atual"])


def adicionar(ini: float, fim: float):
    return _acao(rv.adicionar, ini, fim, S["duracao"], S["cfg"]["revisao"]["duracao_min_s"])


def selecionar_rally(evt: gr.SelectData):
    return _selecionar(S["ordem"][evt.index[0]])


def selecionar_lacuna(evt: gr.SelectData):
    r = S["lac"].iloc[evt.index[0]]
    S["t_lacuna"] = float(r.t_pico)
    c = S["cfg"]["revisao"]
    return _player(r.t_pico - c["contexto_antes_s"] - 4, r.t_pico + c["contexto_depois_s"] + 8), max(r.t_pico - 6, r.ini_s), min(r.t_pico + 10, r.fim_s)


def marcar_lacuna():
    if "t_lacuna" in S:
        rv.marcar_lacuna(S["pasta"], S["t_lacuna"])
    return _atualizar()


def cortar():
    r = rv.ativos(S["df"])
    destino = cortar_video(S["bruto"], r, S["pasta"] / f"{S['bruto'].stem}_revisado.mp4", S["cfg"], progresso=lambda *_: None)
    return f"Vídeo gerado: {destino} ({len(r)} rallies, {r['duracao_s'].sum() / 60:.1f} min)"


def exportar(nome: str):
    pend, lac_pend = rv.pendencias(S["df"], S["lac"], rv.marcas_lacunas(S["pasta"]))
    if pend or lac_pend:
        return f"Ainda faltam {pend} rallies e {lac_pend} vazios suspeitos. Revise tudo antes de exportar como dado de treino."
    destino = Path("data/revisados") / nome
    if destino.exists() or (Path("outputs/pares") / nome).exists():
        return f"Já existe um par com o nome {nome}. Escolha outro nome."
    d = S["cfg"]["deteccao"]
    rv.exportar_treino(S["df"], destino, S["bruto"], S["duracao"], d["margem_pre_s"], d["margem_pos_s"])
    return f"Exportado em {destino}. Entra no próximo treino (listar_pares)."


def interface() -> gr.Blocks:
    with gr.Blocks(title="rallycut: revisão") as app:
        gr.Markdown("## Revisão dos cortes")
        with gr.Row():
            corte = gr.Dropdown(_pastas(), label="Corte (outputs/cortes/)", scale=2)
            bruto = gr.Textbox(label="Vídeo bruto (vazio = o do cortar.py)", scale=3)
            modalidade = gr.Radio(["quadra", "praia"], label="Modalidade (vazio = a do cortar.py)", scale=1)
            botao = gr.Button("Carregar", variant="primary", scale=1)
        resumo = gr.Markdown()
        with gr.Row():
            with gr.Column(scale=3):
                with gr.Tab("Rallies (menos confiantes primeiro)"):
                    tabela = gr.Dataframe(interactive=False, label="Clique numa linha para assistir")
                with gr.Tab("Possíveis rallies perdidos"):
                    lacunas = gr.Dataframe(interactive=False, label="Clique para assistir ao redor do pico")
                    marcar = gr.Button("Não há rally aqui (marcar como revisado)")
            with gr.Column(scale=4):
                titulo = gr.Markdown("Selecione um rally ou um vazio.")
                video = gr.HTML()
                with gr.Row():
                    ini = gr.Number(label="Início (s)")
                    fim = gr.Number(label="Fim (s)")
                with gr.Row():
                    ok = gr.Button("Aceitar e próximo", variant="primary")
                    ajusta = gr.Button("Salvar ajuste")
                    tira = gr.Button("Remover rally", variant="stop")
                    nova = gr.Button("Adicionar rally com estes tempos")
        with gr.Row():
            nome = gr.Textbox(label="Nome do par (modalidade_nome)", scale=2)
            exporta = gr.Button("Exportar como dado de treino", scale=1)
            corta = gr.Button("Cortar o vídeo revisado", scale=1)
        msg = gr.Markdown()

        saidas = [tabela, lacunas, resumo]
        botao.click(carregar, [corte, bruto, modalidade], [*saidas, nome])
        tabela.select(selecionar_rally, None, [video, ini, fim, titulo])
        lacunas.select(selecionar_lacuna, None, [video, ini, fim])
        ok.click(aceitar_e_proximo, None, [*saidas, video, ini, fim, titulo])
        ajusta.click(ajustar, [ini, fim], saidas)
        tira.click(remover, None, saidas)
        nova.click(adicionar, [ini, fim], saidas)
        marcar.click(marcar_lacuna, None, saidas)
        exporta.click(exportar, nome, msg)
        corta.click(cortar, None, msg)
    return app


if __name__ == "__main__":
    interface().launch(allowed_paths=[str(CORTES.resolve())], inbrowser=False)
