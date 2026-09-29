"""Fatia a arte gerada em camadas registradas.

Roda o FaceLandmarker na propria arte para descobrir onde estao olhos, boca
e queixo: assim os recortes saem alinhados sem ninguem medir pixel na mao, e
as variantes (olhos fechados, bocas) recortam a MESMA caixa da base, o que
garante registro perfeito na hora de trocar o sprite.

Uso:  .venv\\Scripts\\python.exe tools/fatiar.py
Saida: art/layers/*.png + art/avatar.json
"""

import json
import sys
from pathlib import Path

import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision
from PIL import Image, ImageChops, ImageDraw, ImageFilter

RAIZ = Path(__file__).resolve().parent.parent
ART = RAIZ / "art"
OUT = ART / "layers"

# indices de landmark do FaceLandmarker (canonical face mesh)
OLHO_E = [33, 133, 159, 145, 160, 144, 158, 153]
OLHO_D = [362, 263, 386, 374, 385, 380, 387, 373]
SOBR_E = [70, 63, 105, 66, 107]
SOBR_D = [300, 293, 334, 296, 336]
BOCA = [61, 291, 0, 17, 13, 14, 78, 308, 82, 87]
QUEIXO = 152
ROSTO_OVAL = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397,
              365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58,
              132, 93, 234, 127, 162, 21, 54, 103, 67, 109]


def landmarks(caminho):
    opts = vision.FaceLandmarkerOptions(
        base_options=mp_python.BaseOptions(
            model_asset_path=str(RAIZ / "models" / "face_landmarker.task")
        ),
        running_mode=vision.RunningMode.IMAGE,
        num_faces=1,
    )
    with vision.FaceLandmarker.create_from_options(opts) as det:
        img = Image.open(caminho).convert("RGBA")
        # o modelo nao entende alfa: compoe sobre cinza medio antes de detectar
        fundo = Image.new("RGBA", img.size, (128, 128, 128, 255))
        fundo.alpha_composite(img)
        rgb = np.ascontiguousarray(np.asarray(fundo.convert("RGB")))
        res = det.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))
    if not res.face_landmarks:
        raise SystemExit(f"Nenhum rosto encontrado em {caminho}")
    w, h = img.size
    return [(p.x * w, p.y * h) for p in res.face_landmarks[0]], img.size


def caixa(pts, indices, margem_x, margem_y, tamanho):
    xs = [pts[i][0] for i in indices]
    ys = [pts[i][1] for i in indices]
    x0 = max(0, min(xs) - margem_x)
    x1 = min(tamanho[0], max(xs) + margem_x)
    y0 = max(0, min(ys) - margem_y)
    y1 = min(tamanho[1], max(ys) + margem_y)
    return tuple(int(v) for v in (x0, y0, x1, y1))


def recortar(origem, box, destino):
    Image.open(origem).convert("RGBA").crop(box).save(destino)


def main():
    base = ART / "base.png"
    if not base.exists():
        raise SystemExit(f"Falta a arte base em {base}")
    pts, size = landmarks(base)
    OUT.mkdir(parents=True, exist_ok=True)

    b_olho_e = caixa(pts, OLHO_E, 26, 24, size)
    b_olho_d = caixa(pts, OLHO_D, 26, 24, size)
    b_sobr_e = caixa(pts, SOBR_E, 22, 20, size)
    b_sobr_d = caixa(pts, SOBR_D, 22, 20, size)
    b_boca = caixa(pts, BOCA, 46, 42, size)
    queixo_y = int(pts[QUEIXO][1])

    manifesto = {
        "base": "base.png",
        "size": list(size),
        "corte_pescoco": queixo_y + 30,
        "caixas": {
            "olho_e": b_olho_e, "olho_d": b_olho_d,
            "sobr_e": b_sobr_e, "sobr_d": b_sobr_d,
            "boca": b_boca,
        },
        "camadas": {},
    }

    img_base = Image.open(base).convert("RGBA")

    # Corpo e a arte inteira; a cabeca e a MESMA arte recortada por uma elipse
    # de borda suave em volta do rosto e do cabelo.
    #
    # Um corte horizontal levava junto o topo do capuz, e qualquer movimento da
    # cabeca deixava uma emenda reta visivel no ombro. Com a elipse, o que se
    # move e so a cabeca, e o corpo intacto por baixo preenche o resto - a
    # borda difusa esconde o encontro das duas camadas.
    img_base.save(OUT / "corpo.png")
    manifesto["camadas"]["corpo"] = {"arquivo": "corpo.png", "pos": [0, 0]}

    oval = [pts[i] for i in ROSTO_OVAL]
    cx = sum(x for x, _ in oval) / len(oval)
    topo = min(y for _, y in oval)
    base_y = max(y for _, y in oval)
    largura = max(x for x, _ in oval) - min(x for x, _ in oval)
    # margem generosa: o cabelo sobe bem acima do oval do rosto
    rx = largura * 0.95
    ry = (base_y - topo) * 0.82
    cy = (topo + base_y) / 2 - ry * 0.12

    mascara = Image.new("L", size, 0)
    ImageDraw.Draw(mascara).ellipse(
        [cx - rx, cy - ry, cx + rx, cy + ry + ry * 0.35], fill=255
    )
    mascara = mascara.filter(ImageFilter.GaussianBlur(14))

    cabeca = img_base.copy()
    alfa = ImageChops.multiply(cabeca.split()[3], mascara)
    cabeca.putalpha(alfa)
    cabeca.save(OUT / "cabeca.png")
    manifesto["camadas"]["cabeca"] = {"arquivo": "cabeca.png", "pos": [0, 0]}
    manifesto["elipse_cabeca"] = [cx, cy, rx, ry]

    # sprites de troca: mesma caixa, arquivos diferentes
    variantes = {
        "olhos_abertos": (base, [("olho_e", b_olho_e), ("olho_d", b_olho_d)]),
        "olhos_fechados": (ART / "olhos_fechados.png",
                           [("olho_e", b_olho_e), ("olho_d", b_olho_d)]),
        "boca_fechada": (base, [("boca", b_boca)]),
        "boca_a": (ART / "boca_a.png", [("boca", b_boca)]),
        "boca_i": (ART / "boca_i.png", [("boca", b_boca)]),
        "boca_o": (ART / "boca_o.png", [("boca", b_boca)]),
    }

    faltando = []
    for nome, (origem, pecas) in variantes.items():
        if not Path(origem).exists():
            faltando.append(nome)
            continue
        for peca, box in pecas:
            arq = f"{nome}__{peca}.png"
            recortar(origem, box, OUT / arq)
            manifesto["camadas"][f"{nome}:{peca}"] = {
                "arquivo": arq, "pos": [box[0], box[1]]
            }

    (ART / "avatar.json").write_text(json.dumps(manifesto, indent=2), encoding="utf-8")
    print(f"elipse da cabeca: centro=({cx:.0f},{cy:.0f}) raios=({rx:.0f},{ry:.0f})")
    print(f"caixas: olho_e={b_olho_e} boca={b_boca}")
    print(f"{len(manifesto['camadas'])} camadas escritas em {OUT}")
    if faltando:
        print(f"variantes ainda nao geradas: {', '.join(faltando)}", file=sys.stderr)


if __name__ == "__main__":
    main()
