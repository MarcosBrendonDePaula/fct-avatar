"""Landmarks de mao -> pose desenhavel.

O MediaPipe entrega 21 pontos por mao. O avatar nao precisa de 21: precisa
saber ONDE desenhar a mao, de que TAMANHO, com que INCLINACAO e se esta
aberta ou fechada. Este modulo reduz os 21 pontos a esses quatro numeros,
com a mesma suavizacao usada no rosto.
"""

import math

from .filtros import Smoother, clamp

PULSO = 0
POLEGAR_PONTA = 4
INDICADOR_MCP = 5
INDICADOR_PONTA = 8
MEDIO_MCP = 9
MEDIO_PONTA = 12
ANELAR_MCP = 13
MINIMO_MCP = 17
MINIMO_PONTA = 20

PONTAS = (INDICADOR_PONTA, MEDIO_PONTA, 16, MINIMO_PONTA)
BASES = (INDICADOR_MCP, MEDIO_MCP, ANELAR_MCP, MINIMO_MCP)


class PoseMao:
    """Pose suavizada de uma mao, em coordenadas normalizadas 0..1."""

    def __init__(self, lado):
        self.lado = lado
        self.visivel = False
        self.x = 0.5
        self.y = 0.5
        self.escala = 1.0
        self.angulo = 0.0     # graus, 0 = dedos para cima
        self.abertura = 1.0   # 0 punho fechado .. 1 mao aberta
        self.presenca = 0.0   # 0..1, para a mao surgir e sumir sem piscar

        self._s = {}

    def _suave(self, chave, alvo, dt, meia_vida=0.05):
        if chave not in self._s:
            self._s[chave] = Smoother(meia_vida, alvo)
        self._s[chave].half_life = meia_vida
        return self._s[chave](alvo, dt)

    def atualizar(self, mao, dt):
        """`mao` e um Hand do tracker, ou None quando nao ha deteccao."""
        if mao is None:
            # Some por fade, nao por corte. A deteccao pisca em quadros
            # isolados, e sem isso a mao apareceria e sumiria piscando.
            self.presenca = self._suave("presenca", 0.0, dt, 0.12)
            self.visivel = self.presenca > 0.02
            return self

        p = mao.landmarks
        pulso = p[PULSO]
        medio = p[MEDIO_MCP]

        # centro: entre o pulso e a base do dedo medio, que e o centro visual
        # da palma - o pulso sozinho joga a mao para baixo demais
        cx = (pulso[0] + medio[0]) / 2
        cy = (pulso[1] + medio[1]) / 2

        # tamanho: distancia pulso -> base do medio, que quase nao muda com a
        # mao abrindo ou fechando (ao contrario da distancia ate as pontas)
        palma = math.hypot(medio[0] - pulso[0], medio[1] - pulso[1])

        # inclinacao: mesma direcao, medida como angulo de tela
        ang = math.degrees(math.atan2(medio[0] - pulso[0], pulso[1] - medio[1]))

        # abertura: pontas longe da palma = aberta. Normalizado pelo tamanho
        # da propria palma, senao afastar a mao da camera "fecharia" o punho.
        if palma > 1e-5:
            espalhamento = sum(
                math.hypot(p[t][0] - cx, p[t][1] - cy) for t in PONTAS
            ) / (4 * palma)
        else:
            espalhamento = 1.6
        abertura = clamp((espalhamento - 1.05) / 0.85, 0.0, 1.0)

        self.presenca = self._suave("presenca", 1.0, dt, 0.10)
        self.x = self._suave("x", cx, dt, 0.045)
        self.y = self._suave("y", cy, dt, 0.045)
        self.escala = self._suave("escala", palma, dt, 0.08)
        self.abertura = self._suave("abertura", abertura, dt, 0.05)
        # angulo entra sem suavizacao circular: a mao nao passa de +-180 na
        # pratica, e tratar a volta complicaria mais do que resolve
        self.angulo = self._suave("angulo", ang, dt, 0.06)
        self.visivel = True
        return self


class RastreioMaos:
    """Mantem uma PoseMao por lado, casando deteccoes entre quadros."""

    def __init__(self):
        self.esquerda = PoseMao("Left")
        self.direita = PoseMao("Right")

    def atualizar(self, maos, dt):
        por_lado = {}
        for m in maos or []:
            # fica com a deteccao de maior confianca por lado
            atual = por_lado.get(m.side)
            if atual is None or m.confidence > atual.confidence:
                por_lado[m.side] = m
        self.esquerda.atualizar(por_lado.get("Left"), dt)
        self.direita.atualizar(por_lado.get("Right"), dt)
        return [p for p in (self.esquerda, self.direita) if p.visivel]
