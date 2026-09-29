"""Renderer baseado nos PNGs recortados de art/.

Interface: render(AvatarParams) -> imagem BGR (numpy), pronta para exibir.

Tudo acontece em numpy/cv2, em BGR. A versao anterior ia e voltava entre
PIL e numpy a cada frame e compunha camadas do tamanho do quadro inteiro;
custava 10,5 ms e limitava o desenho a ~47 fps. As tres mudancas que
importam, todas medidas:

1. O corpo e o fundo sao compostos UMA vez, no construtor. Por frame so
   sobra copiar esse plano pronto.
2. As camadas ficam recortadas na caixa util (a cabeca ocupa metade do
   quadro), entao a rotacao e a mistura tocam bem menos pixels.
3. Nada de PIL no caminho quente: converter RGBA->BGR custava 2,6 ms por
   frame sozinho e agora nem existe, porque as camadas ja sao carregadas
   em BGR.
"""

import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

RAIZ = Path(__file__).resolve().parent.parent
ART = RAIZ / "art"

ENTRA_GIRO = 0.45     # a partir daqui troca para a arte de 3/4
VOLTA_FRENTE = 0.30   # e so volta a frontal abaixo daqui (histerese)
MARGEM_GIRO = 90      # folga em volta da cabeca para a rotacao nao cortar

# viseme -> sprite disponivel (e/u reaproveitam os vizinhos mais proximos)
VISEME_SPRITE = {
    "fechada": "boca_fechada", "a": "boca_a", "i": "boca_i",
    "e": "boca_i", "o": "boca_o", "u": "boca_o",
}


def _carregar_bgra(caminho, escala):
    im = Image.open(caminho).convert("RGBA")
    if escala != 1.0:
        im = im.resize((max(1, round(im.width * escala)),
                        max(1, round(im.height * escala))), Image.LANCZOS)
    return cv2.cvtColor(np.asarray(im), cv2.COLOR_RGBA2BGRA)


def _caixa_util(bgra):
    """Menor retangulo que contem tudo que nao e transparente."""
    ys, xs = np.nonzero(bgra[:, :, 3] > 2)
    if len(xs) == 0:
        return 0, 0, bgra.shape[1], bgra.shape[0]
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def compor(fundo_bgr, src_bgra, x, y):
    """Compoe src sobre fundo_bgr no ponto (x, y). Altera fundo_bgr."""
    h, w = src_bgra.shape[:2]
    H, W = fundo_bgr.shape[:2]
    x0, y0 = max(0, x), max(0, y)
    x1, y1 = min(W, x + w), min(H, y + h)
    if x0 >= x1 or y0 >= y1:
        return
    src = src_bgra[y0 - y:y1 - y, x0 - x:x1 - x]
    roi = fundo_bgr[y0:y1, x0:x1]
    # Escrito com as operacoes do OpenCV, que sao SIMD. A mesma conta em
    # numpy com uint16 custava 21 ms por frame - sozinha, o gargalo inteiro
    # do desenho.
    a3 = cv2.cvtColor(src[:, :, 3], cv2.COLOR_GRAY2BGR)
    frente = cv2.multiply(src[:, :, :3], a3, scale=1 / 255.0)
    fundo = cv2.multiply(roi, cv2.bitwise_not(a3), scale=1 / 255.0)
    cv2.add(frente, fundo, dst=roi)


class SpriteRenderer:
    def __init__(self, saida=(720, 720), bg=(0, 0, 0, 0), manifesto=None):
        caminho = Path(manifesto or ART / "avatar.json")
        if not caminho.exists():
            raise FileNotFoundError(
                f"Falta {caminho}. Rode:  .venv\\Scripts\\python.exe tools/fatiar.py"
            )
        self.m = json.loads(caminho.read_text(encoding="utf-8"))
        self.size = saida
        self.escala = saida[1] / self.m["size"][1]
        self._vista = None
        self._faltando = set()

        base_dir = ART / "layers"
        cam = {}
        for nome, info in self.m["camadas"].items():
            arq = base_dir / info["arquivo"]
            if arq.exists():
                cam[nome] = (_carregar_bgra(arq, self.escala),
                             (round(info["pos"][0] * self.escala),
                              round(info["pos"][1] * self.escala)))

        # plano de fundo + corpo: estatico, composto uma unica vez
        w, h = saida
        self.plano = np.empty((h, w, 3), np.uint8)
        self.plano[:] = bg[:3]
        corpo = cam.pop("corpo", None)
        if corpo is not None:
            compor(self.plano, corpo[0], *corpo[1])

        # cabecas recortadas na caixa util, com folga para a rotacao
        self.cabecas = {}
        x0, y0, x1, y1 = _caixa_util(cam["cabeca"][0])
        self.hx0 = max(0, x0 - MARGEM_GIRO)
        self.hy0 = max(0, y0 - MARGEM_GIRO)
        hx1 = min(w, x1 + MARGEM_GIRO)
        hy1 = min(h, y1 + MARGEM_GIRO)
        for nome in ("cabeca", "cabeca_esq", "cabeca_dir"):
            if nome in cam:
                self.cabecas[nome] = np.ascontiguousarray(
                    cam[nome][0][self.hy0:hy1, self.hx0:hx1]
                )

        # sprites de olho e boca, em coordenadas da caixa da cabeca
        self.pecas = {}
        for nome, (img, (px, py)) in cam.items():
            if nome.startswith("cabeca"):
                continue
            self.pecas[nome] = (img, (px - self.hx0, py - self.hy0))

        self.pivo = (self.cabecas["cabeca"].shape[1] / 2,
                     self.m["corte_pescoco"] * self.escala - self.hy0)

    # ------------------------------------------------------------------ util
    def _peca(self, nome):
        p = self.pecas.get(nome)
        if p is None and nome not in self._faltando:
            self._faltando.add(nome)
            print(f"[sprite] camada ausente: {nome}")
        return p

    def _colar(self, alvo, nome):
        p = self._peca(nome)
        if p is not None:
            self._compor_bgra(alvo, p[0], p[1])

    def _mesclar(self, alvo, nome_a, nome_b, t):
        """Cola a mistura de dois sprites da MESMA caixa (t=0 -> a, t=1 -> b)."""
        pa, pb = self._peca(nome_a), self._peca(nome_b)
        if pa is None or pb is None:
            self._colar(alvo, nome_b if pa is None else nome_a)
            return
        mix = cv2.addWeighted(pa[0], 1.0 - t, pb[0], t, 0)
        self._compor_bgra(alvo, mix, pa[1])

    @staticmethod
    def _compor_bgra(alvo, src, pos):
        """Compoe src (BGRA) sobre alvo (BGRA) preservando o alfa do alvo."""
        x, y = pos
        h, w = src.shape[:2]
        H, W = alvo.shape[:2]
        x0, y0 = max(0, x), max(0, y)
        x1, y1 = min(W, x + w), min(H, y + h)
        if x0 >= x1 or y0 >= y1:
            return
        s = src[y0 - y:y1 - y, x0 - x:x1 - x]
        roi = alvo[y0:y1, x0:x1]
        a3 = cv2.cvtColor(s[:, :, 3], cv2.COLOR_GRAY2BGR)
        cor = cv2.add(cv2.multiply(s[:, :, :3], a3, scale=1 / 255.0),
                      cv2.multiply(roi[:, :, :3], cv2.bitwise_not(a3),
                                   scale=1 / 255.0))
        roi[:, :, :3] = cor
        np.maximum(roi[:, :, 3], s[:, :, 3], out=roi[:, :, 3])

    def _decidir_vista(self, turn):
        if self._vista is None:
            if abs(turn) > ENTRA_GIRO:
                self._vista = "cabeca_dir" if turn > 0 else "cabeca_esq"
        else:
            girando_dir = self._vista == "cabeca_dir"
            if abs(turn) < VOLTA_FRENTE or (turn > 0) != girando_dir:
                self._vista = None
        return self._vista

    # --------------------------------------------------------------- publico
    def render(self, p):
        # Giro lateral: PNG 2D nao gira em 3D, entao ha artes em 3/4 e a vista
        # e TROCADA, nao misturada. Cruzar as duas sobrepunha dois narizes e
        # duas bocas. Misturar so serve entre imagens quase iguais, como olho
        # aberto e fechado.
        vista = self._decidir_vista(p.head_turn)
        if vista and vista in self.cabecas:
            cabeca = self.cabecas[vista].copy()
        else:
            cabeca = self.cabecas["cabeca"].copy()

            # Olhos: com dois quadros de arte, o meio-termo sai de uma mistura.
            # Espremer o sprite aberto arrastava cabelo e sobrancelha junto.
            for lado, ab in (("olho_e", p.eye_open_l), ("olho_d", p.eye_open_r)):
                if ab < 0.12:
                    self._colar(cabeca, f"olhos_fechados:{lado}")
                elif ab > 0.88:
                    self._colar(cabeca, f"olhos_abertos:{lado}")
                else:
                    self._mesclar(cabeca, f"olhos_fechados:{lado}",
                                  f"olhos_abertos:{lado}", ab)
            self._colar(cabeca, f"{VISEME_SPRITE.get(p.viseme, 'boca_fechada')}:boca")

        # Amplitudes limitadas de proposito: o furo aberto na camada do corpo
        # e menor que a cabeca, e essa folga e o que impede a fresta de
        # aparecer. Mexer mais que isso comeca a mostrar o buraco.
        bob = (p.bounce - 0.5) * 4
        dx = p.head_x * 28 + p.head_turn * 10
        dy = p.head_y * 22 + bob

        m = cv2.getRotationMatrix2D(self.pivo, -p.head_tilt * 9, 1.0)
        m[0, 2] += dx
        m[1, 2] += dy
        cabeca = cv2.warpAffine(cabeca, m, (cabeca.shape[1], cabeca.shape[0]),
                                flags=cv2.INTER_LINEAR,
                                borderMode=cv2.BORDER_CONSTANT,
                                borderValue=(0, 0, 0, 0))

        quadro = self.plano.copy()
        compor(quadro, cabeca, self.hx0, self.hy0)
        return quadro
