# rallycut

Corte automático de rallies em vídeos de vôlei (quadra e praia) gravados com câmera
fixa em tripé. Recebe o vídeo bruto e entrega um vídeo só com os rallies, mais um
CSV/JSON com os trechos e os capítulos para a descrição do YouTube.

**Prioridade máxima:** nunca perder o início de um rally. Sobrar um pouco de tempo
morto é aceitável; cortar o saque não é.

## Status

| Fase | Descrição | Situação |
|------|-----------|----------|
| 0 | Ambiente e baseline de desempenho | em andamento |
| 1 | Dataset a partir dos vídeos já editados | pendente |
| 2 | Detector v1 (regras calibradas) | pendente |
| 3 | App Gradio com revisão | pendente |
| 4 | Classificador v2 | pendente |

## Requisitos

- Windows com GPU NVIDIA (testado em RTX 3050 Laptop, 4 GB de VRAM)
- [uv](https://docs.astral.sh/uv/) (gerencia o Python 3.12 e as dependências)
- ffmpeg com `h264_nvenc` e `cuda` (`winget install Gyan.FFmpeg`)

## Instalação

```bash
uv sync --all-groups
uv run pytest
```

PyTorch é instalado com CUDA 13.0 pelo índice `pytorch-cu130` (ver `pyproject.toml`).

## Estrutura

```
src/rallycut/     código da pipeline (config, proxy, features, detector, corte)
app/              app Gradio (Fase 3)
configs/          configuração em YAML (margens, fps do proxy, limiares, saída)
scripts/          scripts de linha de comando (benchmark, avaliação, ...)
tests/            testes unitários
data/             vídeos e rótulos (fora do git)
outputs/          arquivos gerados (fora do git)
```

Vídeos e dados **nunca** entram no repositório: `data/` e `outputs/` estão no `.gitignore`,
assim como as extensões de vídeo e áudio.

## Configuração

Tudo é configurável em `configs/default.yaml`: fps e altura do proxy, resolução e
codec de saída, margens pré/pós, limiares e peso do áudio. A seção `geral` vale para
tudo e as seções `quadra` e `praia` a sobrescrevem.

## Fase 0: baseline de desempenho

```bash
uv run python scripts/phase0_benchmark.py --video data/Teste1.mp4 --modalidade quadra
```

Mede o tempo de gerar o proxy (decodificação NVDEC), extrair o áudio e codificar a
saída 1080p (NVENC), além do pico de VRAM, e extrapola para um jogo de 2 h. O
relatório fica em `outputs/phase0/baseline.json`.

Resultado de referência (`Teste1.mp4`: 4K HEVC, 59 fps, 254 s):

| Etapa | Tempo | x tempo real | Estimativa para 2 h |
|-------|-------|--------------|---------------------|
| Proxy 480p, 8 fps | 46,3 s | 5,5x | ~22 min |
| Áudio mono 16 kHz | 0,6 s | 411x | ~0,3 min |
| Saída 1080p NVENC | 65,2 s | 3,9x | ~31 min |
| **Total (sem análise)** | **112 s** | | **~53 min** |

O gargalo é a decodificação do 4K HEVC a 59 fps. Os seus jogos são 30 fps, então a
estimativa real tende a ser menor. O pico extra de VRAM ficou em ~0,5 GB, bem abaixo dos 4 GB.
