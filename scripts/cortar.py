"""Corta os rallies de um vídeo bruto de vôlei.

Gera em outputs/cortes/<nome>/: rallies.csv (início e fim de cada rally, editável) e <nome>_cortado.mp4.
Tudo que é caro fica em cache na mesma pasta (rodar de novo só refaz o que falta).
Para revisar: edite rallies.csv (colunas rally, ini_s, fim_s) e rode com --cortar-de.

Uso:
    uv run python scripts/cortar.py VIDEO_BRUTO --modalidade quadra|praia
    uv run python scripts/cortar.py VIDEO_BRUTO --modalidade praia --cortar-de outputs/cortes/jogo/rallies.csv
    --sem-video   só gera o rallies.csv
"""

import argparse
import json
from pathlib import Path

import pandas as pd

from rallycut.config import MODALIDADES, carregar_config
from rallycut.corte import cortar_video
from rallycut.inferencia import probabilidades, rallies


def principal():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("video", type=Path)
    ap.add_argument("--modalidade", required=True, choices=MODALIDADES)
    ap.add_argument("--saida", type=Path, default=None, help="pasta de saída (padrão: outputs/cortes/<nome do vídeo>)")
    ap.add_argument("--modelo", type=Path, default=Path("outputs/modelos/tcn_bola.pt"))
    ap.add_argument("--mapa", type=Path, default=Path("outputs/bola/mapa_todos.pt"), help="detector de bola")
    ap.add_argument("--cortar-de", type=Path, default=None, help="usa este rallies.csv (revisado) em vez de detectar")
    ap.add_argument("--sem-video", action="store_true", help="só gera o rallies.csv")
    args = ap.parse_args()

    saida = args.saida or Path("outputs/cortes") / args.video.stem
    saida.mkdir(parents=True, exist_ok=True)
    (saida / "fonte.json").write_text(json.dumps({"video": str(args.video.resolve()), "modalidade": args.modalidade}))
    if args.cortar_de:
        r = pd.read_csv(args.cortar_de)
    else:
        p = probabilidades(args.video, saida, args.modalidade, args.modelo, args.mapa)
        r = rallies(p, args.modalidade)
        r.to_csv(saida / "rallies.csv", index=False)
        print(f"{len(r)} rallies, {r['duracao_s'].sum() / 60:.1f} min -> {saida / 'rallies.csv'}")
    if not args.sem_video:
        destino = cortar_video(args.video, r, saida / f"{args.video.stem}_cortado.mp4", carregar_config(args.modalidade))
        print(f"vídeo: {destino}")


if __name__ == "__main__":
    principal()
