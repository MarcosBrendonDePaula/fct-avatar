"""Tracking com MediaPipe: rosto (landmarks + 52 blendshapes + matriz de pose)
e maos. Publica tudo em AvatarFrame, mesmo o que o avatar ainda nao usa.
"""

import math
import time
from pathlib import Path

import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

from .types import AvatarFrame, Hand, HeadPose

MODELS = Path(__file__).resolve().parent.parent / "models"

ESPELHO_LADO = {"Left": "Right", "Right": "Left"}


def _euler_from_matrix(m):
    """Extrai yaw/pitch/roll (graus) da matriz 4x4 de transformacao facial.

    A matriz e R = Rz*Ry*Rx, com X para a direita, Y para cima e Z para a
    camera. Logo o que interessa para uma cabeca e:

        yaw   (virar para os lados) = rotacao em Y
        pitch (acenar cima/baixo)   = rotacao em X
        roll  (tombar a cabeca)     = rotacao em Z

    Eu tinha atribuido os tres na ordem errada, e dava para ver: passando a
    arte com a cabeca virada de lado pelo tracker, o giro saia no canal do
    acenar.
    """
    r = np.asarray(m, dtype=float)[:3, :3]
    cy = math.sqrt(r[2, 1] ** 2 + r[2, 2] ** 2)
    yaw = math.atan2(-r[2, 0], cy)
    if cy > 1e-6:
        pitch = math.atan2(r[2, 1], r[2, 2])
        roll = math.atan2(r[1, 0], r[0, 0])
    else:
        pitch = math.atan2(-r[1, 2], r[1, 1])
        roll = 0.0
    # pitch sai negativo quando a cabeca sobe; invertemos para casar com o
    # contrato de HeadPose ("+ = olhando para cima"). Conferido passando as
    # artes olha_cima/olha_baixo pelo proprio tracker.
    return math.degrees(yaw), -math.degrees(pitch), math.degrees(roll)


class Tracker:
    def __init__(self, track_hands=True, hand_every=3):
        # maos custam ~3x o rosto e se movem devagar em relacao a face:
        # rodamos a cada N frames e reaproveitamos o ultimo resultado.
        self.hand_every = max(1, hand_every)
        self._n = 0
        self._last_hands = []

        face_opts = vision.FaceLandmarkerOptions(
            base_options=mp_python.BaseOptions(
                model_asset_path=str(MODELS / "face_landmarker.task")
            ),
            running_mode=vision.RunningMode.VIDEO,
            num_faces=1,
            output_face_blendshapes=True,
            output_facial_transformation_matrixes=True,
        )
        self.face = vision.FaceLandmarker.create_from_options(face_opts)

        self.hand = None
        if track_hands:
            hand_opts = vision.HandLandmarkerOptions(
                base_options=mp_python.BaseOptions(
                    model_asset_path=str(MODELS / "hand_landmarker.task")
                ),
                running_mode=vision.RunningMode.VIDEO,
                num_hands=2,
            )
            self.hand = vision.HandLandmarker.create_from_options(hand_opts)

        self._t0 = time.time()

    def process(self, frame_bgr):
        """frame BGR -> AvatarFrame."""
        rgb = frame_bgr[:, :, ::-1]
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb))
        ts_ms = int((time.time() - self._t0) * 1000)

        out = AvatarFrame(timestamp=time.time())

        face_res = self.face.detect_for_video(image, ts_ms)
        if face_res.face_landmarks:
            out.face_present = True
            lms = face_res.face_landmarks[0]

            if face_res.face_blendshapes:
                out.blendshapes = {
                    c.category_name: c.score for c in face_res.face_blendshapes[0]
                }

            yaw = pitch = roll = 0.0
            if face_res.facial_transformation_matrixes:
                yaw, pitch, roll = _euler_from_matrix(
                    face_res.facial_transformation_matrixes[0]
                )

            out.face_points = [(p.x, p.y) for p in lms]

            xs = [p.x for p in lms]
            ys = [p.y for p in lms]
            width = max(xs) - min(xs)
            out.head = HeadPose(
                yaw=yaw,
                pitch=pitch,
                roll=roll,
                x=sum(xs) / len(xs),
                y=sum(ys) / len(ys),
                scale=width / 0.25 if width else 1.0,
            )

        if self.hand is not None:
            self._n += 1
            if self._n % self.hand_every == 0:
                hand_res = self.hand.detect_for_video(image, ts_ms)
                maos = []
                for i, lms in enumerate(hand_res.hand_landmarks or []):
                    handedness = hand_res.handedness[i][0]
                    maos.append(
                        Hand(
                            # O quadro chega espelhado (capture.py inverte para
                            # que mexer para a direita mova para a direita), mas
                            # o MediaPipe decide qual mao e qual supondo imagem
                            # NAO espelhada. Os rotulos vem trocados, e sem
                            # desfazer isso a arte da mao sai virada ao
                            # contrario.
                            side=ESPELHO_LADO.get(handedness.category_name,
                                                  handedness.category_name),
                            confidence=handedness.score,
                            landmarks=[(p.x, p.y, p.z) for p in lms],
                        )
                    )
                self._last_hands = maos
            out.hands = self._last_hands

        return out

    def close(self):
        self.face.close()
        if self.hand is not None:
            self.hand.close()
