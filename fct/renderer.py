"""Renderer 2D em camadas.

O v1 usa um avatar placeholder desenhado em codigo (aparece na hora, sem
depender de arte pronta). A interface publica e render(params) -> RGBA, e
e por tras dela que entram, depois, os PNGs do personagem real e a
deformacao estilo Live2D.
"""

import cv2
import numpy as np
from PIL import Image, ImageDraw


def _rotacionar(img, graus, center):
    """Rotacao via OpenCV: ~10x mais barata que PIL.rotate bicubico neste
    tamanho, e e o custo dominante do frame."""
    if abs(graus) < 0.25:
        return img
    arr = np.asarray(img)
    m = cv2.getRotationMatrix2D(center, graus, 1.0)
    out = cv2.warpAffine(arr, m, (img.width, img.height),
                         flags=cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))
    return Image.fromarray(out, "RGBA")

W, H = 720, 720

PELE = (255, 222, 200, 255)
PELE_SOMBRA = (236, 194, 172, 255)
CABELO = (66, 52, 92, 255)
CABELO_LUZ = (96, 78, 130, 255)
ROUPA = (58, 122, 168, 255)
ROUPA_ESC = (44, 96, 134, 255)
OLHO_BRANCO = (252, 252, 255, 255)
IRIS = (72, 148, 196, 255)
PUPILA = (28, 32, 48, 255)
BOCA_INT = (150, 62, 74, 255)
LINHA = (58, 44, 66, 255)
BLUSH = (255, 150, 155, 70)


class Renderer:
    """Desenha o avatar. Sem estado entre frames: params entram, imagem sai."""

    def __init__(self, size=(W, H), bg=(0, 0, 0, 0)):
        self.size = size
        self.bg = bg
        self._corpo = None   # o corpo so se desloca: desenha uma vez, reusa

    def render(self, p):
        canvas = Image.new("RGBA", self.size, self.bg)
        w, h = self.size

        bob = (p.bounce - 0.5) * 6            # respiracao idle
        hx = p.head_x * 55
        hy = p.head_y * 45 + bob

        self._draw_corpo(canvas, p, hx * 0.35, hy * 0.3)

        head = self._draw_cabeca(p)
        head = _rotacionar(head, -p.head_tilt * 14,
                           center=(head.width / 2, head.height * 0.92))
        cx = int(w / 2 - head.width / 2 + hx)
        cy = int(h * 0.09 + hy)
        canvas.alpha_composite(head, (cx, cy))
        return canvas

    # ------------------------------------------------------------------ corpo
    def _draw_corpo(self, canvas, p, dx, dy):
        if self._corpo is None:
            w, h = self.size
            corpo = Image.new("RGBA", (w, h + 120), (0, 0, 0, 0))
            d = ImageDraw.Draw(corpo)
            cx, top = w / 2, h * 0.66
            d.rounded_rectangle([cx - 150, top, cx + 150, h + 60], radius=70, fill=ROUPA)
            d.rounded_rectangle([cx - 150, top, cx - 60, h + 60], radius=70, fill=ROUPA_ESC)
            d.rounded_rectangle([cx - 38, top - 70, cx + 38, top + 24], radius=26,
                                fill=PELE_SOMBRA)
            self._corpo = corpo
        # paste (nao alpha_composite) porque aceita offset negativo, e o corpo
        # e a primeira coisa desenhada num canvas ainda vazio.
        canvas.paste(self._corpo, (int(dx), int(dy)), self._corpo)

    # ----------------------------------------------------------------- cabeca
    def _draw_cabeca(self, p):
        layer = Image.new("RGBA", (620, 640), (0, 0, 0, 0))
        d = ImageDraw.Draw(layer)
        cx, cy = 310, 300
        turn = p.head_turn          # -1..1
        nod = p.head_nod
        sx = turn * 34              # deslocamento das feicoes pela rotacao
        sy = -nod * 26

        # cabelo de tras (fica atras de tudo)
        d.ellipse([cx - 196, cy - 206, cx + 196, cy + 150], fill=CABELO)
        for side in (-1, 1):
            hx = cx + side * 176
            d.ellipse([hx - 42, cy - 130, hx + 42, cy + 200], fill=CABELO)

        # orelhas
        for side in (-1, 1):
            ex = cx + side * 166 - sx * 0.35
            d.ellipse([ex - 24, cy - 14, ex + 24, cy + 44], fill=PELE_SOMBRA)

        # rosto
        d.ellipse([cx - 168, cy - 196, cx + 168, cy + 210], fill=PELE)

        # translucidos precisam de camada propria (ImageDraw substitui o pixel
        # em vez de compor), mas a camada e do tamanho da forma, nao do frame.
        if abs(turn) > 0.05:
            s = -1 if turn > 0 else 1   # sombra no lado que "afastou"
            x0 = cx + s * 92
            self._blob(layer, (x0 - 84, cy - 190, x0 + 84, cy + 205), (46, 30, 60, 20))

        # franja: desenhada antes das feicoes, cobre so a testa
        self._draw_franja(d, cx, cy, sx, sy)

        self._draw_olhos(d, cx + sx, cy + sy, p)
        self._draw_sobrancelhas(d, cx + sx, cy + sy, p)
        self._draw_boca(d, cx + sx * 0.85, cy + sy * 0.8, p)

        # blush (translucido)
        for side in (-1, 1):
            bx = cx + sx * 0.9 + side * 100
            self._blob(layer, (bx - 30, cy + 64 + sy, bx + 30, cy + 90 + sy), BLUSH)

        return layer

    @staticmethod
    def _blob(layer, box, cor):
        """Elipse translucida composta corretamente, numa camada do tamanho
        da propria forma (evita alocar um RGBA inteiro por frame)."""
        x0, y0, x1, y1 = (int(v) for v in box)
        w, h = x1 - x0, y1 - y0
        if w <= 0 or h <= 0:
            return
        patch = Image.new("RGBA", (w, h), (0, 0, 0, 0))
        ImageDraw.Draw(patch).ellipse([0, 0, w - 1, h - 1], fill=cor)
        layer.alpha_composite(patch, dest=(max(0, x0), max(0, y0)))

    def _draw_olhos(self, d, cx, cy, p):
        ey = cy + 10
        for side, openness in ((-1, p.eye_open_l), (1, p.eye_open_r)):
            ex = cx + side * 74
            rw, rh_full = 46, 34
            rh = max(1.5, rh_full * max(0.0, min(1.0, openness)))

            if openness < 0.08:
                d.line([ex - rw, ey, ex + rw, ey], fill=LINHA, width=7)
                continue

            d.ellipse([ex - rw, ey - rh, ex + rw, ey + rh], fill=OLHO_BRANCO)
            # iris segue o olhar, limitada ao branco do olho
            gx = p.gaze_x * 18
            gy = p.gaze_y * -12
            ir = min(26, rh * 1.5 + 6)
            d.ellipse([ex + gx - ir, ey + gy - ir, ex + gx + ir, ey + gy + ir], fill=IRIS)
            pr = ir * 0.5
            d.ellipse([ex + gx - pr, ey + gy - pr, ex + gx + pr, ey + gy + pr], fill=PUPILA)
            # brilho
            d.ellipse([ex + gx + 4, ey + gy - ir * 0.7, ex + gx + 4 + 9,
                       ey + gy - ir * 0.7 + 9], fill=(255, 255, 255, 235))
            # contorno / palpebra
            d.arc([ex - rw, ey - rh, ex + rw, ey + rh], 180, 360, fill=LINHA, width=6)

    def _draw_sobrancelhas(self, d, cx, cy, p):
        for side, lift in ((-1, p.brow_l), (1, p.brow_r)):
            bx = cx + side * 74
            by = cy - 44 - lift * 22
            tilt = -lift * 8 * side
            d.line([bx - 42, by + 6 + tilt, bx + 42, by - 4 - tilt],
                   fill=CABELO, width=11)

    def _draw_boca(self, d, cx, cy, p):
        my = cy + 122
        open_h = p.mouth_open * 46
        wide = 1.0 + p.mouth_wide * 0.45
        v = p.viseme

        if v == "fechada":
            d.arc([cx - 34 * wide, my - 16, cx + 34 * wide, my + 18],
                  0, 180, fill=LINHA, width=6)
            return

        if v == "u":
            rw, rh = 20, max(12, open_h * 0.8)
        elif v == "o":
            rw, rh = 30, max(20, open_h)
        elif v == "i":
            rw, rh = 48 * wide, max(8, open_h * 0.45)
        elif v == "e":
            rw, rh = 46 * wide, max(14, open_h * 0.8)
        else:  # "a"
            rw, rh = 36 * wide, max(16, open_h)

        d.ellipse([cx - rw, my - rh * 0.6, cx + rw, my + rh], fill=BOCA_INT)
        d.ellipse([cx - rw, my - rh * 0.6, cx + rw, my + rh], outline=LINHA, width=5)

    def _draw_franja(self, d, cx, cy, sx, sy):
        """Calota + franja. Fica acima dos olhos (que estao em cy+10)."""
        f = sx * 0.5  # a franja acompanha de leve o giro da cabeca
        # calota
        d.pieslice([cx - 176, cy - 204, cx + 176, cy + 30], 180, 360, fill=CABELO)
        # franja em mechas, terminando pouco acima dos olhos
        d.polygon([(cx - 176, cy - 120), (cx - 168 + f, cy - 196),
                   (cx - 40 + f, cy - 206), (cx - 62 + f, cy - 62)], fill=CABELO)
        d.polygon([(cx - 62 + f, cy - 62), (cx - 40 + f, cy - 206),
                   (cx + 56 + f, cy - 204), (cx + 30 + f, cy - 78)], fill=CABELO_LUZ)
        d.polygon([(cx + 176, cy - 126), (cx + 168 + f, cy - 198),
                   (cx + 56 + f, cy - 204), (cx + 30 + f, cy - 70)], fill=CABELO)
