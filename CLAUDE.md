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
- `treino.py` (dados e métricas), `temporal.py` (TCN), `bola.py` (detector de bola por mapa de calor + features de bola).
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
- **Bola (já na main; tag `base-8b247c7` = antes da bola)**: `bola.py` = mapa de calor da bola (MobileNetV3, CenterNet) treinado com 300 cliques (`bola_amostrar/rotular/treinar.py`; COCO falha em bola tricolor: recall 0-12% em 2 jogos). Jogo de fora: recall em rally 72-90% (score 0,3). Candidatos a 8 fps/1280x720 (`bola_extrair.py`, ~4-7 min/vídeo, detector sem o jogo avaliado), bolas paradas descartadas, 7 features/janela (`features_bola`, `com_bola`). AUC só com bola: praia 0,90 (atuais 0,77; set2 0,68 -> 0,96), quadra 0,92. TCN por vídeo+início, 3 sementes, 1,5x: **praia 78% -> 90±2%** (todos os 5 jogos melhoram: +5 a +19), quadra 95% = 95%; controle (bola deslocada no tempo) 79% = sem ganho. Auditoria (`bola_auditoria.py`): inícios que ainda se perdem = a bola só aparece ~2 s depois do início (saque de longe, bola pequena) ou pessoa passa na frente da câmera; set2 71% (12/38), set4 86%. Ressalva: as features de treino de cada jogo vêm de detectores que viram frames (sem rótulo de rally) do jogo de teste. TCN final com bola: `uv run python scripts/treinar_tcn_final.py` -> `outputs/modelos/tcn_bola.pt` (53 features; o `tcn.pt` antigo, sem bola, ficou ao lado). Vídeo novo: `extrair_candidatos` com `outputs/bola/mapa_todos.pt` (`carregar_mapa`) -> `features_bola` -> `prever_video`. Falta: `cortar.py` de ponta a ponta (bruto -> features -> bola -> rallies -> corte NVENC).
- Novos pares de quadra (outro confronto, `nduvsmedunimes`, bruto 1080p): `quadra_set3` alinhou bem e, com jogo de fora (TCN+início+bola, 8 pares), preserva **93%** dos inícios (quadra média 95%, praia 88%): a quadra generaliza para esse jogo. `quadra_set4` teve o editado refeito (o primeiro tinha enquadramento diferente e o alinhamento falhou); realinhado (41 rallies, custo médio 0,056). Com 9 pares (TCN+início+bola, jogo de fora, 3 sementes, 1,5x): **quadra 95%** (set1 93, set2 98, set3 92, set4 94) e **praia 92%** (set1 94, set2 84, set3 93, set4 92, Set5 95). `tcn_bola.pt` retreinado com os 9 pares (`tcn_bola_anterior.pt` = versão com 7). Pasta com `IGNORAR` é pulada por `listar_pares`.
- **`cortar.py` pronto** (`inferencia.py`, `corte.py`; modelo `tcn_bola.pt` guarda colunas e PCA). Limiares calibrados com jogo de fora (`calibrar_limiar.py`): quadra 0,40 (97%|pior 94%, 1,8x), praia 0,25 (97%|95%, 2,3x), margem pré 4 s. Verificado em vídeo novo (modelo e detector sem o jogo): quadra_set3 94% (1 rally inteiro perdido, 1 começando 6,6 s tarde; 1,87x), praia_set4 97% (1 rally perdido; 2,35x); folga antes do saque nos preservados >= 4,9 s; vídeo cortado coerente (duração = soma dos rallies +0,4 a +1,1 s, 1080p, áudio e vídeo em sincronia, emendas sem falhas); ~14 min por vídeo de 17-18 min. Perdas são rallies que o detector não viu (não o saque cortado). Para retreinar: `treinar_tcn_final.py`, depois `calibrar_limiar.py` e ajustar `configs/default.yaml`.
- Próximos passos possíveis: mais pares (praia por posição de câmera; quadra de outro ginásio), proxy único por vídeo para acelerar, Fase 3 (app Gradio com revisão; praia semi-automática), corte final com NVENC. Capítulos do YouTube: adiado pelo usuário.
