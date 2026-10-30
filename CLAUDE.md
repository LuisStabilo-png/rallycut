# rallycut

Pipeline local para cortar rallies de vídeos de vôlei (quadra/praia, câmera fixa).
Código, comentários e README em português.

## Regras do projeto
- **Prioridade máxima:** recall alto nos inícios de rally. Nunca cortar o saque; tempo morto sobrando é aceitável.
- Antes de executar cada fase, apresentar o plano detalhado e esperar aprovação do usuário.
- Antes de apagar, sobrescrever ou renomear arquivo existente, mostrar o que muda e esperar confirmação.
- Downloads (vídeos, pesos de modelos) só com confirmação (arquivo, origem, tamanho); pesos em `data/modelos/`, hash conferido.
- Não modificar nada fora da pasta do projeto. Commits pequenos e descritivos ao fim de cada etapa.
- Tudo configurável em `configs/default.yaml` (margens, fps do proxy, limiares, resolução de saída).
- Vídeos e dados nunca entram no git (`data/`, `outputs/`).
- Código enxuto, sem código morto. O usuário quer respostas curtas e economia de tokens.

## Ambiente
- Windows, RTX 3050 Laptop (4 GB de VRAM), 7,7 GB de RAM: processar em streaming, evitar vários modelos na GPU.
- Sempre rodar via `uv run` (não há `python` no PATH). ffmpeg com NVDEC e NVENC.
- O bruto de celular tem fps variável: buscar frames por TEMPO (`CAP_PROP_POS_MSEC`), nunca por índice.

## Como o projeto funciona
Dataset = **pares** em `data/pares/<quadra|praia>_<nome>/` (um `*bruto*` e um `*editado*`; o editado só remove trechos).
- `alinhamento.py`: alinha editado ao bruto (descritores sem fundo + Viterbi + correlação de áudio) -> `trechos.csv` (rallies) e `removidos.csv` (tempo morto). É o gabarito.
- `features.py`: por janela de 0,5 s do bruto: movimento, jogadores (torchvision Faster R-CNN), áudio e embedding visual (MobileNetV3). Cache em `outputs/pares/<par>/janelas_<altura>.csv` e `embeddings_<altura>.npy` (360p = sem sufixo; padrão atual 720).
- `deteccao.py`: contexto temporal, `segmentar` (probabilidades -> rallies) e `avaliar` (% de inícios preservados, perdidos, falsos, tempo morto sobrando).
- `treino.py` (dados e métricas), `temporal.py` (TCN).
- `cortes.py`/`pessoas.py`: detector de cortes para vídeos já editados (Fase 1, método de reserva).

## Comandos
```bash
uv sync --all-groups
uv run pytest
uv run python scripts/treinar.py            # alinha/extrai o que faltar e treina o LightGBM (sem gastar tokens)
uv run python scripts/treinar_temporal.py   # compara LightGBM x TCN, jogo de fora
uv run python scripts/fase2_estatisticas.py --pasta outputs/pares/quadra_set1
```

## Estado e achados (2026-10-30)
- 6 pares: quadra_set1/2 (mesmo ginásio e câmera), praia_set1..4 (câmeras em posições diferentes).
- Estatísticas da quadra: rally mediana 10 s; tempo morto removido >= ~15 s (unir trechos com gap < 10 s é seguro).
- Quadra com LightGBM + embeddings: 86-100% dos inícios preservados (jogo de fora 89-93%) com ~40-50% de tempo morto sobrando (saída ~14 min vs 8,4 do gabarito). Config de recall alto (limiar 0,2, margem pré 4 s) em `configs/default.yaml`. Generalização para OUTRO ginásio ainda não testada.
- Praia: não utilizável (AUC jogo de fora 0,65-0,86; só chega a bons inícios mantendo 2-3x o vídeo ideal). Cada jogo tem posição de câmera diferente. Resolução de detecção 720p vs 360p: sem ganho. Normalizar features por vídeo: sem ganho claro.
- Áudio não separa rally de tempo morto (toques/s iguais dentro e fora); detector de toques foi removido.
- Descartado: compensação de movimento de câmera, candidatos só-áudio, filtro `media_min`.
- Pendente: experimento `treinar_temporal.py` (TCN global/por vídeo vs LightGBM, jogo de fora, vídeo final = 1,5x o gabarito) foi disparado mas o resultado não foi lido. Rodar de novo e ler a tabela.
- Próximos passos possíveis: mais pares (praia por posição de câmera; quadra de outro ginásio), proxy único por vídeo para acelerar, Fase 3 (app Gradio com revisão; praia semi-automática), corte final com NVENC. Capítulos do YouTube: adiado pelo usuário.
