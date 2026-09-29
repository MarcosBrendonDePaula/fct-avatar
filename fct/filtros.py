"""Filtros de sinal compartilhados.

Ficam fora de state.py porque o rastreio de maos usa os mesmos, e ter os
dois modulos se importando fechava um ciclo de import.
"""

import math

def clamp(v, lo=-1.0, hi=1.0):
    return max(lo, min(hi, v))


def remap(v, in_lo, in_hi, out_lo=0.0, out_hi=1.0):
    if in_hi == in_lo:
        return out_lo
    t = (v - in_lo) / (in_hi - in_lo)
    return out_lo + clamp(t, 0.0, 1.0) * (out_hi - out_lo)


def deadzone(v, dz=0.04):
    if abs(v) < dz:
        return 0.0
    return (abs(v) - dz) / (1 - dz) * (1 if v > 0 else -1)


class Smoother:
    """Filtro exponencial com meia-vida em segundos (estavel a fps variavel)."""

    def __init__(self, half_life=0.06, initial=0.0):
        self.half_life = half_life
        self.value = initial

    def __call__(self, target, dt):
        if dt <= 0:
            return self.value
        alpha = 1.0 - 0.5 ** (dt / self.half_life)
        self.value += (target - self.value) * alpha
        return self.value
