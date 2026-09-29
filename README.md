# Avatar virtual com face tracking

Avatar 2D estilo VTuber que segue seu rosto pela webcam.

## Instalacao

Precisa de **Python 3.12** (o MediaPipe ainda nao publica wheels para 3.13+;
se o seu Python do sistema for mais novo, instale o 3.12 ao lado dele, com
`winget install Python.Python.3.12` no Windows).

```
py -3.12 -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe tools/baixar_modelos.py
```

A arte fica em `art/` e as camadas recortadas sao geradas a partir dela:

```
.venv\Scripts\python.exe tools/fatiar.py
```

## Rodar

```
.venv\Scripts\python.exe preview.py --auto-brilho
```

Abre duas janelas: **Avatar** (limpa, e a que o OBS deve capturar) e
**Avatar - debug** (sua camera com a malha do rosto e os parametros).

| tecla | acao |
|---|---|
| C | calibra a pose de repouso (fique neutro e aperte) |
| A | liga/desliga o ajuste automatico de brilho |
| + / - | ganho manual (quarto escuro) |
| [ / ] | gama: levanta as sombras sem estourar as luzes |
| M | mostra/esconde a malha do rosto |
| D | mostra/esconde a janela de debug |
| P | alterna entre a arte e o boneco placeholder |
| Q / ESC | sai |

Opcoes: `--camera 1`, `--no-hands` (mais fps), `--fps 60`, `--ganho 1.8`,
`--gama 0.7`, `--placeholder`.

## O que funciona

Cabeca (girar, inclinar, acenar, deslocar), com artes em 3/4 para os lados;
piscada por olho com meio-termo; direcao do olhar; sobrancelhas; abertura de
boca e visema (a/i/u/e/o); respiracao idle; retorno suave ao neutro quando o
rosto sai de quadro; correcao de brilho para ambiente escuro.

**O que ainda nao funciona:** o personagem nao tem bracos nem maos na arte
(o enquadramento e de busto), entao as maos sao rastreadas mas nao animam
nada. Nas vistas de 3/4, olhos e boca ficam parados, porque essas vistas
ainda nao tem sprites proprios. Os visemas E e U reaproveitam I e O.

## Estrutura

| arquivo | papel |
|---|---|
| `fct/capture.py` | webcam, em thread propria (Media Foundation) |
| `fct/tracker.py` | MediaPipe -> AvatarFrame (dados crus, completos) |
| `fct/state.py` | AvatarFrame -> AvatarParams (suavizado, calibrado) |
| `fct/sprite_renderer.py` | AvatarParams -> imagem BGR |
| `fct/renderer.py` | boneco placeholder, util sem arte |
| `fct/pipeline.py` | tracking em thread, desacoplado do desenho |
| `fct/imagem.py` | ganho/gama do frame antes do tracking |
| `tools/fatiar.py` | corta a arte em camadas registradas |
| `preview.py` | laco principal e janelas |

Design e decisoes:
`docs/superpowers/specs/2026-09-28-avatar-face-tracking-design.md`
