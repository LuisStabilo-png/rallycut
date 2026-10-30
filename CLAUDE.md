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
- 7 pares: quadra_set1/2 (mesmo ginásio e câmera), praia_set1..4 e praia_Set5 (câmeras em posições diferentes; a pasta se chama `praia_Set5`, com S maiúsculo).
- Estatísticas da quadra: rally mediana 10 s; tempo morto removido >= ~15 s (unir trechos com gap < 10 s é seguro).
- Quadra com LightGBM + embeddings: 86-100% dos inícios preservados (jogo de fora 89-93%) com ~40-50% de tempo morto sobrando (saída ~14 min vs 8,4 do gabarito). Config de recall alto (limiar 0,2, margem pré 4 s) em `configs/default.yaml`. Generalização para OUTRO ginásio ainda não testada.
- Praia: não utilizável (AUC jogo de fora 0,65-0,86; só chega a bons inícios mantendo 2-3x o vídeo ideal). Cada jogo tem posição de câmera diferente. Resolução de detecção 720p vs 360p: sem ganho. Normalizar features por vídeo: sem ganho claro.
- Áudio não separa rally de tempo morto (toques/s iguais dentro e fora); detector de toques foi removido.
- Descartado: compensação de movimento de câmera, candidatos só-áudio, filtro `media_min`.
- TCN x LightGBM (`treinar_temporal.py --sementes 3`, 7 pares, jogo de fora, saída = 1,5x o gabarito, média ± desvio): início preservado médio quadra 89% (LightGBM) / 93±2% (TCN global) / 93±1% (TCN por vídeo); praia 64% / 65±3% / **75±1%**. TCN por vídeo é o melhor na praia (set1 57% -> 76%, set3 50% -> 78%, Set5 80% -> 89%), exceto praia_set2 (~55%, AUC 0,68 em todos). Quadra: TCN ≥ LightGBM, global ≈ por vídeo. Praia ainda abaixo do ideal.
- Histerese na segmentação (`histerese` 0,5) e ensemble (média de percentis LightGBM + TCNs), mesmo protocolo, 3 sementes: sem ganho claro. Ensemble sobe a AUC (praia 0,73-0,93) mas o início preservado fica 94% quadra / 71% praia (TCN por vídeo sozinho: 93% / 75%). Histerese: TCN praia +1 a +3 pts, LightGBM e ensemble pioram; dentro do ruído. Mantida desligada (1.0). Melhor modelo segue sendo o TCN por vídeo.
- TCN com 2ª saída (entorno do início, −3 a +2 s) e probabilidade = max(rally, início), mesmo protocolo, 3 sementes: praia global 65% -> 74%, por vídeo 76% -> **78±1%**; quadra 95% (igual). Ganho pequeno mas consistente na praia (set1 81->87, set3 82->85, Set5 83->86); set2 segue ~60% (AUC 0,67). Melhor modelo atual: TCN por vídeo com saída de início.
- TCN final salvo: `scripts/treinar_tcn_final.py` -> `outputs/modelos/tcn.pt` (por vídeo, 2 saídas, 3 sementes); uso: `carregar_tcn` + `prever_video` em `temporal.py` (entrada: features brutas na ordem de `colunas`). Autotreino no vídeo novo (ajuste fino com palpites confiantes p>0,8 / p<0,1): sem ganho (praia 78% -> 78±3%, set2 60% -> 57%); código removido.
- Próximos passos possíveis: mais pares (praia por posição de câmera; quadra de outro ginásio), proxy único por vídeo para acelerar, Fase 3 (app Gradio com revisão; praia semi-automática), corte final com NVENC. Capítulos do YouTube: adiado pelo usuário.
