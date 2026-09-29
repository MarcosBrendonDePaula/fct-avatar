"""Ajuste do frame da camera antes do tracking.

Num quarto escuro o MediaPipe perde o rosto muito antes de a imagem ficar
ilegivel para o olho humano: o detector depende de contraste local. Clarear
o frame ANTES do tracking costuma valer mais que qualquer ajuste depois.
"""

import cv2
import numpy as np


class Ajuste:
    def __init__(self, ganho=1.0, gama=1.0, auto=False, alvo=115):
        self.ganho = ganho      # multiplica o brilho
        self.gama = gama        # <1 clareia as sombras sem estourar as luzes
        self.auto = auto        # persegue o brilho medio `alvo`
        self.alvo = alvo
        self.brilho = 0.0       # ultimo brilho medido, para o painel
        self._lut_gama = None
        self._lut_para = None

    def _lut(self, gama):
        if self._lut_para != gama:
            g = max(0.05, gama)
            self._lut_gama = np.array(
                [((i / 255.0) ** (1.0 / g)) * 255 for i in range(256)]
            ).astype(np.uint8)
            self._lut_para = gama
        return self._lut_gama

    def aplicar(self, frame):
        # luminancia media por amostragem: medir o frame inteiro nao muda a
        # decisao e custaria mais que o proprio ajuste
        amostra = frame[::8, ::8]
        self.brilho = float(amostra.mean())

        if self.auto:
            # passo suave: corrigir de uma vez faz a imagem pulsar
            desejado = self.alvo / max(8.0, self.brilho)
            self.ganho += (min(4.0, desejado) - self.ganho) * 0.12

        out = frame
        if abs(self.ganho - 1.0) > 0.01:
            out = cv2.convertScaleAbs(out, alpha=self.ganho, beta=0)
        if abs(self.gama - 1.0) > 0.01:
            out = cv2.LUT(out, self._lut(self.gama))
        return out

    def resumo(self):
        modo = "auto" if self.auto else "manual"
        return f"ganho {self.ganho:.2f} gama {self.gama:.2f} ({modo})"
