"""Miniaturas antes/depois dos cortes para validação visual."""

import base64
from pathlib import Path

import cv2
import numpy as np


def par_de_miniaturas(video: Path | str, frame: int, largura: int = 480) -> np.ndarray:
    """Imagem lado a lado: último frame antes do corte (esquerda) e primeiro depois (direita).

    `frame` é o índice do primeiro frame após o corte (coluna `frame` de cortes.csv)."""
    cap = cv2.VideoCapture(str(video))
    try:
        cap.set(cv2.CAP_PROP_POS_FRAMES, max(frame - 1, 0))
        ok_a, antes = cap.read()
        ok_d, depois = cap.read()
    finally:
        cap.release()
    if not (ok_a and ok_d):
        raise RuntimeError(f"não consegui ler os frames {frame - 1}/{frame} de {video}")
    altura = round(antes.shape[0] * largura / antes.shape[1])
    antes, depois = (cv2.resize(f, (largura, altura), interpolation=cv2.INTER_AREA)
                     for f in (antes, depois))
    return np.hstack([antes, np.full((altura, 4, 3), 255, np.uint8), depois])


def _jpeg_base64(imagem: np.ndarray, qualidade: int = 70) -> str:
    ok, buf = cv2.imencode(".jpg", imagem, [cv2.IMWRITE_JPEG_QUALITY, qualidade])
    return base64.b64encode(buf.tobytes()).decode("ascii")


def _frame(video: Path, frame: int, largura: int) -> np.ndarray:
    cap = cv2.VideoCapture(str(video))
    try:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame)
        ok, img = cap.read()
    finally:
        cap.release()
    if not ok:
        raise RuntimeError(f"não consegui ler o frame {frame} de {video}")
    return cv2.resize(img, (largura, round(img.shape[0] * largura / img.shape[1])),
                      interpolation=cv2.INTER_AREA)


def _mmss(t: float) -> str:
    return f"{int(t // 60):02d}:{t % 60:04.1f}"


def gerar_html(video: Path, cortes, trechos, fps: float, destino: Path,
               n_amostra_alta: int = 20, n_amostra_media: int = 40, longo_s: float = 45.0,
               curto_s: float = 2.0, semente: int = 0, progresso=print) -> None:
    """Relatório HTML autocontido: todos os cortes de confiança baixa, uma amostra dos de
    confiança média e alta, e os trechos suspeitos (muito curtos ou muito longos)."""
    rng = np.random.default_rng(semente)

    def amostrar(conf: str, n: int):
        g = cortes[cortes["confianca"] == conf]
        return g.iloc[np.sort(rng.choice(len(g), min(n, len(g)), replace=False))]

    outros = cortes[~cortes["confianca"].isin(["alta", "media"])]
    amostra = amostrar("alta", n_amostra_alta)
    selecionados = np.concatenate([outros["id"].to_numpy(), amostra["id"].to_numpy(),
                                   amostrar("media", n_amostra_media)["id"].to_numpy()])
    sel = cortes[cortes["id"].isin(selecionados)].sort_values("tempo_s")

    cartoes = []
    for i, c in enumerate(sel.itertuples(), 1):
        img = par_de_miniaturas(video, int(c.frame), 360)
        estado = "aceito" if c.aceito else "rejeitado"
        cartoes.append(
            f'<div class="card {c.confianca} {estado}"><img src="data:image/jpeg;base64,{_jpeg_base64(img)}">'
            f'<div><b>#{c.id}</b> {_mmss(c.tempo_s)} &middot; <span class="tag {c.confianca}">{c.confianca}</span> '
            f'&middot; {estado}<br>z visual {c.z_visual:.0f} &middot; z áudio {c.z_audio:.0f} '
            f'&middot; origem {c.origem}</div></div>')
        if i % 10 == 0:
            progresso(f"  miniaturas {i}/{len(sel)}")

    suspeitos = []
    for t in trechos.itertuples():
        if t.duracao_s >= longo_s:
            pontos = np.linspace(t.inicio_s, t.fim_s, 7)[1:-1]
            tiras = "".join(f'<img class="tira" src="data:image/jpeg;base64,'
                            f'{_jpeg_base64(_frame(video, int(p * fps), 240))}">' for p in pontos)
            suspeitos.append(f'<div class="susp"><b>Rally {t.rally}</b> {_mmss(t.inicio_s)}–{_mmss(t.fim_s)} '
                             f'({t.duracao_s:.0f} s): possível corte perdido. Há salto de cena na tira?<br>{tiras}</div>')
        elif t.duracao_s < curto_s:
            suspeitos.append(f'<div class="susp"><b>Rally {t.rally}</b> {_mmss(t.inicio_s)}–{_mmss(t.fim_s)} '
                             f'({t.duracao_s:.1f} s): muito curto, possível corte falso '
                             f'(veja os cortes vizinhos).</div>')

    n_ac = int(cortes["aceito"].sum())
    html = f"""<!doctype html><html lang="pt-BR"><meta charset="utf-8">
<title>Validação de cortes: {video.stem}</title>
<style>
body{{font-family:system-ui,sans-serif;margin:16px;background:#fafafa;color:#222}}
.grade{{display:flex;flex-wrap:wrap;gap:10px}}
.card{{background:#fff;border:2px solid #ccc;border-radius:6px;padding:6px;font-size:13px;width:732px;max-width:100%}}
.card img{{width:100%;display:block;margin-bottom:4px}}
.card.baixa{{border-color:#d9534f}}.card.media{{border-color:#f0ad4e}}.card.alta{{border-color:#5cb85c}}
.tag{{padding:1px 6px;border-radius:4px;color:#fff;font-size:12px}}
.tag.baixa{{background:#d9534f}}.tag.media{{background:#f0ad4e}}.tag.alta{{background:#5cb85c}}
.susp{{background:#fff;border:1px solid #bbb;padding:6px;margin:6px 0;font-size:13px}}
.tira{{height:90px;margin:2px}}
code{{background:#eee;padding:1px 4px}}
</style>
<h1>Validação de cortes: {video.name}</h1>
<p>{len(cortes)} candidatos, <b>{n_ac} aceitos</b> ({len(trechos)} rallies). Cada cartão mostra o último frame
<b>antes</b> (esquerda) e o primeiro <b>depois</b> (direita) do corte. Um corte real mostra os jogadores em
outras posições.</p>
<p>Anote os erros em <code>correcoes.csv</code> (colunas <code>acao,valor</code>):
<code>remover,&lt;id&gt;</code> (não é corte), <code>aceitar,&lt;id&gt;</code> (é corte, mas estava rejeitado),
<code>adicionar,&lt;segundos&gt;</code> (corte perdido).</p>
<h2>Suspeitos ({len(suspeitos)})</h2>{''.join(suspeitos) or '<p>Nenhum.</p>'}
<h2>Cortes para validar ({len(sel)}: todos os de baixa confiança + amostras dos médios e altos)</h2>
<div class="grade">{''.join(cartoes)}</div></html>"""
    destino.write_text(html, encoding="utf-8")


def rotular(imagem: np.ndarray, texto: str) -> np.ndarray:
    """Escreve `texto` no canto superior esquerdo, com fundo escuro para leitura."""
    img = imagem.copy()
    (w, h), _ = cv2.getTextSize(texto, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    cv2.rectangle(img, (0, 0), (w + 8, h + 10), (0, 0, 0), -1)
    cv2.putText(img, texto, (4, h + 3), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return img
