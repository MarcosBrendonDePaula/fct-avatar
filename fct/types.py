"""Contratos de dados entre os estagios do pipeline.

AvatarFrame  = saida crua do tracker (publica TUDO, mesmo o que o avatar
               ainda nao consome: olhar, sobrancelhas, maos).
AvatarParams = saida do estagio de estado, ja suavizada, no vocabulario
               do avatar. O renderer so conhece esta estrutura.
"""

from dataclasses import dataclass, field


@dataclass
class HeadPose:
    yaw: float = 0.0     # graus, + = olhando para a direita da tela
    pitch: float = 0.0   # graus, + = olhando para cima
    roll: float = 0.0    # graus, + = cabeca tombada para a direita
    x: float = 0.5       # centro do rosto, normalizado 0..1
    y: float = 0.5
    scale: float = 1.0   # proporcional a distancia da camera


@dataclass
class Hand:
    side: str                      # "Left" | "Right"
    confidence: float
    landmarks: list                # 21 tuplas (x, y, z) normalizadas


@dataclass
class AvatarFrame:
    timestamp: float
    face_present: bool = False
    head: HeadPose = field(default_factory=HeadPose)
    blendshapes: dict = field(default_factory=dict)  # 52 coeficientes MediaPipe
    hands: list = field(default_factory=list)
    face_points: list = field(default_factory=list)  # 468 landmarks normalizados,
                                                     # so para visualizacao/debug


@dataclass
class AvatarParams:
    """Parametros estaveis de avatar. Todos em faixas previsiveis."""

    head_turn: float = 0.0    # -1..1
    head_tilt: float = 0.0    # -1..1
    head_nod: float = 0.0     # -1..1
    head_x: float = 0.0       # -1..1, deslocamento lateral
    head_y: float = 0.0       # -1..1
    bounce: float = 0.0       # respiracao/idle, 0..1

    eye_open_l: float = 1.0   # 0 fechado .. 1 aberto
    eye_open_r: float = 1.0
    gaze_x: float = 0.0       # -1..1
    gaze_y: float = 0.0
    brow_l: float = 0.0       # -1 franzida .. 1 levantada
    brow_r: float = 0.0

    mouth_open: float = 0.0   # 0..1
    mouth_wide: float = 0.0   # 0..1
    smile: float = 0.0        # 0..1, sorriso separado da fala
    viseme: str = "fechada"   # fechada|sorriso|sorriso_aberto|a|i|u|e|o

    maos: list = field(default_factory=list)  # PoseMao visiveis neste quadro

    present: bool = False     # rosto detectado neste frame
