"""Camada de imagem no formato que o renderer consome.

Cor e transparencia ficam em DOIS arrays contiguos, nao num BGRA unico:

    cor  - BGR ja multiplicado pelo alfa (pre-multiplicado)
    inv  - (255 - alfa) em 3 canais

Parece redundante, mas foi medido. Guardando BGRA, cada composicao pagava
4,8 ms so para fatiar `bgra[:, :, :3]` (numpy nao consegue dar essa vista ao
OpenCV sem copiar) e mais 3,8 ms para expandir o alfa em 3 canais - 9 dos
11,7 ms de uma composicao eram remanejo de memoria, nao conta.

Com a cor pre-multiplicada e o alfa ja invertido e expandido, compor vira
duas operacoes SIMD sobre buffers contiguos:

    destino = cor + destino * inv
"""

import cv2
import numpy as np
from PIL import Image


class Camada:
    __slots__ = ("cor", "inv")

    def __init__(self, cor, inv):
        self.cor = cor
        self.inv = inv

    # ------------------------------------------------------------- criacao
    @classmethod
    def de_arquivo(cls, caminho, escala=1.0):
        im = Image.open(caminho).convert("RGBA")
        if escala != 1.0:
            im = im.resize((max(1, round(im.width * escala)),
                            max(1, round(im.height * escala))), Image.LANCZOS)
        rgba = np.asarray(im)
        bgr = cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGR)
        a3 = cv2.cvtColor(np.ascontiguousarray(rgba[:, :, 3]), cv2.COLOR_GRAY2BGR)
        return cls(cv2.multiply(bgr, a3, scale=1 / 255.0), cv2.bitwise_not(a3))

    @property
    def forma(self):
        return self.cor.shape[:2]

    def recortar(self, y0, y1, x0, x1):
        return Camada(np.ascontiguousarray(self.cor[y0:y1, x0:x1]),
                      np.ascontiguousarray(self.inv[y0:y1, x0:x1]))

    def copia(self, com_alfa=True):
        """Copia a camada. Sem `com_alfa`, o alfa e COMPARTILHADO com a
        original - serve quando so a cor vai mudar, como ao colar sprites
        dentro de uma cabeca que ja e opaca ali."""
        return Camada(self.cor.copy(),
                      self.inv.copy() if com_alfa else self.inv)

    def caixa_util(self):
        """Menor retangulo com algo visivel (inv < 255 quer dizer alfa > 0)."""
        visivel = self.inv[:, :, 0] < 253
        ys, xs = np.nonzero(visivel)
        if len(xs) == 0:
            h, w = self.forma
            return 0, 0, w, h
        return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1

    # ----------------------------------------------------------- operacoes
    def transformar(self, m, dsize):
        """Aplica uma afim. A cor sai com borda preta (nada) e o inverso com
        borda branca (totalmente transparente), que e o mesmo estado."""
        return Camada(
            cv2.warpAffine(self.cor, m, dsize, flags=cv2.INTER_LINEAR,
                           borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0)),
            cv2.warpAffine(self.inv, m, dsize, flags=cv2.INTER_LINEAR,
                           borderMode=cv2.BORDER_CONSTANT,
                           borderValue=(255, 255, 255)))

    def remapear(self, mx, my, manter_alfa=True):
        """Deforma so a cor; o alfa fica como estava.

        Deformar o alfa junto esticava a borda suave do recorte e virava um
        halo escuro ao lado do cabelo. Como a silhueta e presa na borda, o
        alfa de antes continua valendo.
        """
        cor = cv2.remap(self.cor, mx, my, cv2.INTER_LINEAR,
                        borderMode=cv2.BORDER_REPLICATE)
        inv = self.inv if manter_alfa else cv2.remap(
            self.inv, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        return Camada(cor, inv)

    def misturar(self, outra, t, com_alfa=True):
        """Interpola duas camadas da mesma forma (t=0 -> self, t=1 -> outra).

        Sem `com_alfa` o alfa do primeiro e reaproveitado: quando a mistura
        vai ser colada numa area opaca, interpolar a transparencia tambem
        seria trabalho jogado fora.
        """
        return Camada(cv2.addWeighted(self.cor, 1 - t, outra.cor, t, 0),
                      cv2.addWeighted(self.inv, 1 - t, outra.inv, t, 0)
                      if com_alfa else self.inv)

    def sobrepor(self, src, x=0, y=0, alfa=True):
        """Compoe `src` sobre esta camada, em (x, y). Altera esta camada.

        `alfa=False` pula a atualizacao da transparencia: usado quando o
        sprite cai numa area ja opaca, onde o resultado seria identico.
        """
        h, w = src.forma
        H, W = self.forma
        x0, y0 = max(0, x), max(0, y)
        x1, y1 = min(W, x + w), min(H, y + h)
        if x0 >= x1 or y0 >= y1:
            return
        sc = src.cor[y0 - y:y1 - y, x0 - x:x1 - x]
        si = src.inv[y0 - y:y1 - y, x0 - x:x1 - x]
        cor = self.cor[y0:y1, x0:x1]
        inv = self.inv[y0:y1, x0:x1] if alfa else None
        cv2.add(sc, cv2.multiply(cor, si, scale=1 / 255.0), dst=cor)
        if alfa:
            # alfa resultante: 1-(1-as)(1-ad)  ->  inv = inv_s * inv_d
            cv2.multiply(inv, si, dst=inv, scale=1 / 255.0)

    def sobre_fundo(self, fundo_bgr, x=0, y=0):
        """Compoe esta camada sobre um BGR opaco. Altera fundo_bgr."""
        h, w = self.forma
        H, W = fundo_bgr.shape[:2]
        x0, y0 = max(0, x), max(0, y)
        x1, y1 = min(W, x + w), min(H, y + h)
        if x0 >= x1 or y0 >= y1:
            return
        cor = self.cor[y0 - y:y1 - y, x0 - x:x1 - x]
        inv = self.inv[y0 - y:y1 - y, x0 - x:x1 - x]
        roi = fundo_bgr[y0:y1, x0:x1]
        cv2.add(cor, cv2.multiply(roi, inv, scale=1 / 255.0), dst=roi)
