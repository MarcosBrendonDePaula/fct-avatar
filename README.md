# Avatar virtual com face tracking

Avatar 2D estilo VTuber que segue seu rosto pela webcam.

## Rodar o preview

```
.venv\Scripts\python.exe preview.py
```

Teclas: **C** calibra a pose de repouso (fique neutro e aperte) ·
**D** liga/desliga o painel de debug · **Q** ou **ESC** sai.

Opcoes: `--camera 1` (outra webcam) · `--no-hands` (mais fps).

## O que ja funciona

Cabeca (girar, inclinar, acenar, deslocar), piscada por olho, direcao do
olhar, sobrancelhas, abertura de boca e visema (a/i/u/e/o), respiracao idle
e retorno suave ao neutro quando o rosto sai de quadro. As maos ja sao
rastreadas e publicadas, mas o avatar ainda nao as desenha.

O personagem atual e um placeholder desenhado em codigo. Arte real em
camadas e a proxima fase.

## Estrutura

- `fct/capture.py` — webcam, em thread propria
- `fct/tracker.py` — MediaPipe -> AvatarFrame (dados crus, completos)
- `fct/state.py`  — AvatarFrame -> AvatarParams (suavizado, calibrado)
- `fct/renderer.py` — AvatarParams -> imagem RGBA
- `preview.py` — laco principal e janela

Design e decisoes: `docs/superpowers/specs/2026-09-28-avatar-face-tracking-design.md`
