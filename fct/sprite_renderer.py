"""Renderer baseado nos PNGs recortados de art/.

Mesma interface do renderer placeholder: render(AvatarParams) -> imagem RGBA.
O que muda e a origem dos pixels. A cabeca e composta numa camada propria e
so depois transformada, para que girar a cabeca nao gire os ombros.
"""

import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

RAIZ = Path(__file__).resolve().parent.parent
ART = RAIZ / "art"

# viseme -> sprite disponivel (e/u reaproveitam os vizinhos mais proximos)
VISEME_SPRITE = {
    "fechada": "boca_fechada", "a": "boca_a", "i": "boca_i",
    "e": "boca_i", "o": "boca_o", "u": "boca_o",
}


def _carregar(caminho, escala):
    im = Image.open(caminho).convert("RGBA")
    if escala != 1.0:
        im = im.resize((max(1, int(im.width * escala)), max(1, int(im.height * escala))),
                       Image.LANCZOS)
    return im


class SpriteRenderer:
    def __init__(self, saida=(720, 720), bg=(0, 0, 0, 0), manifesto=None):
        caminho = Path(manifesto or ART / "avatar.json")
        if not caminho.exists():
            raise FileNotFoundError(
                f"Falta {caminho}. Rode:  .venv\\Scripts\\python.exe tools/fatiar.py"
            )
        self.m = json.loads(caminho.read_text(encoding="utf-8"))
        self.size = saida
        self.bg = bg

        # a arte e 1024; trabalhar no tamanho de saida evita reescalar por frame
        self.escala = saida[1] / self.m["size"][1]
        base = ART / "layers"
        self.cam = {}
        for nome, info in self.m["camadas"].items():
            arq = base / info["arquivo"]
            if arq.exists():
                self.cam[nome] = (_carregar(arq, self.escala),
                                  (int(info["pos"][0] * self.escala),
                                   int(info["pos"][1] * self.escala)))
        self.corte = int(self.m["corte_pescoco"] * self.escala)
        self._faltando = set()

    # ------------------------------------------------------------------ util
    def _peca(self, nome):
        peca = self.cam.get(nome)
        if peca is None and nome not in self._faltando:
            self._faltando.add(nome)
            print(f"[sprite] camada ausente: {nome}")
        return peca

    def _colar(self, alvo, nome, dx=0, dy=0):
        peca = self._peca(nome)
        if peca is None:
            return
        im, (x, y) = peca
        alvo.alpha_composite(im, dest=(max(0, x + dx), max(0, y + dy)))

    # ---------------------------------------------------------------- cabeca
    def _montar_cabeca(self, p):
        peca = self._peca("cabeca")
        if peca is None:
            raise RuntimeError("camada 'cabeca' ausente; rode tools/fatiar.py")
        cabeca = peca[0].copy()

        # olhos: sprite aberto espremido verticalmente da o meio-termo, o que
        # evita a piscada "liga/desliga" com so dois quadros de arte.
        for lado, abertura in (("olho_e", p.eye_open_l), ("olho_d", p.eye_open_r)):
            if abertura < 0.22:
                self._colar(cabeca, f"olhos_fechados:{lado}")
            elif abertura > 0.92:
                self._colar(cabeca, f"olhos_abertos:{lado}")
            else:
                self._colar(cabeca, f"olhos_fechados:{lado}")
                self._espremer(cabeca, f"olhos_abertos:{lado}", abertura)

        self._colar(cabeca, f"{VISEME_SPRITE.get(p.viseme, 'boca_fechada')}:boca")
        return cabeca

    def _espremer(self, alvo, nome, fator):
        """Cola o sprite comprimido verticalmente, ancorado na base da caixa."""
        peca = self._peca(nome)
        if peca is None:
            return
        im, (x, y) = peca
        h = max(1, int(im.height * fator))
        alvo.alpha_composite(im.resize((im.width, h), Image.BILINEAR),
                             dest=(x, y + im.height - h))

    # --------------------------------------------------------------- publico
    def render(self, p):
        canvas = Image.new("RGBA", self.size, self.bg)
        w, h = self.size

        bob = (p.bounce - 0.5) * 5
        hx = p.head_x * 42
        hy = p.head_y * 34 + bob

        # corpo: acompanha de leve, para o pescoco nao "quebrar"
        corpo = self._peca("corpo")
        if corpo is not None:
            im, (x, y) = corpo
            canvas.paste(im, (int(x + hx * 0.3), int(y + hy * 0.25)), im)

        cabeca = self._montar_cabeca(p)
        arr = np.asarray(cabeca)
        # rotacao + deslocamento numa unica matriz: giro em torno da base do
        # pescoco, nao do centro da imagem.
        m = cv2.getRotationMatrix2D((cabeca.width / 2, self.corte * 0.98),
                                    -p.head_tilt * 12, 1.0)
        m[0, 2] += hx + p.head_turn * 16
        m[1, 2] += hy
        arr = cv2.warpAffine(arr, m, (cabeca.width, cabeca.height),
                             flags=cv2.INTER_LINEAR,
                             borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))
        canvas.alpha_composite(Image.fromarray(arr, "RGBA"), dest=(0, 0))
        return canvas
