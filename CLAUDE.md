# rallycut

Pipeline local para cortar rallies de vídeos de vôlei (quadra/praia, câmera fixa).
Código, comentários e README em português.

## Regras do projeto
- **Prioridade máxima:** recall alto nos inícios de rally. Nunca cortar o saque; tempo morto sobrando é aceitável.
- Antes de executar cada fase, apresentar o plano detalhado e esperar aprovação do usuário.
- Antes de apagar, sobrescrever ou renomear arquivo existente, mostrar o que muda e esperar confirmação.
- Não modificar nada fora da pasta do projeto. Commits pequenos e descritivos ao fim de cada etapa.
- Tudo configurável em `configs/default.yaml` (margens, fps do proxy, limiares, resolução de saída).
- Vídeos e dados nunca entram no git (`data/`, `outputs/`).
- Análise em proxy de baixa resolução/fps + áudio; o bruto só é usado no corte final.

## Ambiente
- Windows, RTX 3050 Laptop (4 GB de VRAM), 7,7 GB de RAM: processar em streaming, evitar vários modelos na GPU.
- Sempre rodar via `uv run` (não há `python` no PATH).
- ffmpeg com NVDEC (`-hwaccel cuda`) e NVENC (`h264_nvenc`).

## Comandos
```bash
uv sync --all-groups
uv run pytest
uv run python scripts/phase0_benchmark.py --video data/Teste1.mp4 --modalidade quadra
```
