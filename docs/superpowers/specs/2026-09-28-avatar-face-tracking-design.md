# Avatar virtual com face tracking — design

Data: 2026-09-28

## Objetivo

Avatar 2D estilo VTuber que acompanha o rosto do usuário em tempo real para
uso em lives. Saída final: webcam virtual utilizável em qualquer app. O v1
entrega um **preview em janela**; a webcam virtual vem na fase seguinte.

## Decisões

- **Tudo em Python** (abordagem B), num interpretador **3.12 dedicado** ao
  projeto — o Python 3.14 do sistema não tem wheels do MediaPipe.
- **Tracking completo desde o v1**: o `AvatarFrame` publica pose da cabeça,
  os 52 blendshapes do MediaPipe (piscada, olhar, sobrancelhas, visemas) e
  os landmarks das mãos, mesmo o que o avatar ainda não desenha. É o que
  permite ligar expressões, olhar e mãos em fases sem refazer o pipeline.
- **Arte**: avatar placeholder desenhado em código no v1, para validar o
  movimento sem depender de arte pronta. Substituído por PNGs em camadas
  descritos num `avatar.json` na fase 2.

## Arquitetura

```
Webcam ─> capture ─> tracker ─> state ─> renderer ─> [preview | virtualcam]
          (thread)   (MediaPipe) (suaviza) (Pillow)
```

| Módulo | Responsabilidade | Depende de |
|---|---|---|
| `capture.py` | frames BGR espelhados, numa thread própria | OpenCV |
| `tracker.py` | frame → `AvatarFrame` (cru, completo) | MediaPipe |
| `state.py` | `AvatarFrame` → `AvatarParams` (suave, calibrado) | nada |
| `renderer.py` | `AvatarParams` → imagem RGBA | Pillow, OpenCV |
| `preview.py` | laço principal + janela de debug | os acima |

O renderer conhece apenas `AvatarParams`, nunca MediaPipe: trocar o tracker
não toca no desenho.

## Suavização (state.py)

Filtro exponencial com **meia-vida em segundos**, não fator fixo por frame —
o comportamento fica igual a 20 ou a 60 fps. Meia-vida por parâmetro: 25ms
para piscada (precisa ser instantânea), 35ms para boca, 60–70ms para cabeça
e sobrancelhas, 250ms para o retorno ao neutro quando o rosto some. Mais
zona morta contra tremor e calibração de repouso (tecla C).

## Desempenho

Medido em RTX 2080 Ti / webcam 1280x720, com rosto + mãos:

| | antes | depois |
|---|---|---|
| captura | 32 ms (bloqueante) | 0,9 ms (thread) |
| tracking | 52 ms | 21 ms (mãos a cada 3 frames) |
| render | 44 ms | 9 ms (rotação via OpenCV, corpo em cache) |
| **total** | **9,4 fps** | **31,8 fps** |

Três otimizações, cada uma atacando um gargalo medido:
1. Captura em thread — a espera pela câmera sobrepõe o tracking.
2. Mãos a cada 3 frames — custam 3x o rosto e se movem devagar.
3. `cv2.warpAffine` no lugar de `PIL.rotate` bicúbico — sozinho, 33 dos 44ms.

## Armadilhas encontradas

- `ImageDraw` com cor semitransparente **substitui** o pixel em vez de compor.
  Translúcidos (blush, sombra) exigem camada própria + `alpha_composite`.
- A ordem das camadas importa: cabelo desenhado depois das feições cobre os
  olhos. Ordem correta: cabelo de trás → rosto → franja → feições.

## Fases seguintes

1. **Webcam virtual** — `pyvirtualcam` sobre o driver do OBS Studio.
2. **Arte real** — PNGs em camadas + `avatar.json` (âncora, ordem, parâmetro
   que controla cada camada). Trocar de personagem vira trocar arquivos.
3. **Expressões e olhar completos** — mais sprites e regras no manifesto.
4. **Mãos e tronco** — consomem `AvatarFrame.hands`, já publicado.
5. **Deformação estilo Live2D** — entra atrás da interface do renderer.
