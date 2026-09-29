"""AvatarFrame cru -> AvatarParams estaveis.

Aqui mora a diferenca entre "tremendo" e "natural": suavizacao exponencial
independente de fps, zona morta, calibracao de repouso e normalizacao.
E tudo calculo puro, entao e a parte mais facil de testar.
"""

import math
import time

from .types import AvatarParams


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


class StateMapper:
    # faixas de entrada -> saida -1..1
    YAW_RANGE = 30.0
    PITCH_RANGE = 25.0
    ROLL_RANGE = 30.0

    def __init__(self):
        self._s = {}
        self._last_t = None
        self._rest = None          # pose de repouso capturada na calibracao
        self._auto_desde = None    # inicio da janela de calibracao automatica
        self._params = AvatarParams()
        self._t_start = time.time()

    def _smooth(self, key, target, dt, half_life=0.06):
        if key not in self._s:
            self._s[key] = Smoother(half_life, target)
        self._s[key].half_life = half_life
        return self._s[key](target, dt)

    def calibrate(self, frame):
        """Define a pose atual como repouso (tecla C no preview)."""
        if frame.face_present:
            self._rest = (frame.head.yaw, frame.head.pitch, frame.head.roll,
                          frame.head.x, frame.head.y)
            self._auto_desde = None

    def _auto_calibrar(self, frame, now):
        """Calibra sozinho depois de um tempo de rosto estavel.

        Sem isso o avatar comeca torto ate a pessoa descobrir a tecla C: a
        pose neutra de cada rosto (e de cada angulo de webcam) nao e zero.
        """
        if self._rest is not None:
            return
        if not frame.face_present:
            self._auto_desde = None
            return
        if self._auto_desde is None:
            self._auto_desde = now
        elif now - self._auto_desde > 1.5:
            self.calibrate(frame)
            print("repouso calibrado automaticamente (tecla C refaz)")

    def update(self, frame):
        # relogio de parede, nao o timestamp do frame: o mesmo AvatarFrame pode
        # ser reaproveitado em varias voltas do laco, e a animacao idle e a
        # suavizacao precisam continuar avancando mesmo assim.
        now = time.time()
        dt = 1 / 30 if self._last_t is None else max(1e-3, now - self._last_t)
        self._last_t = now

        self._auto_calibrar(frame, now)

        p = self._params
        p.present = frame.face_present

        # idle: respiracao suave, roda mesmo sem rosto
        p.bounce = 0.5 + 0.5 * math.sin((now - self._t_start) * 1.6)

        if not frame.face_present:
            # volta suavemente ao neutro em vez de congelar
            for key in ("head_turn", "head_tilt", "head_nod", "head_x", "head_y",
                        "gaze_x", "gaze_y", "brow_l", "brow_r", "mouth_open",
                        "mouth_wide"):
                setattr(p, key, self._smooth(key, 0.0, dt, 0.25))
            p.eye_open_l = self._smooth("eye_l", 1.0, dt, 0.25)
            p.eye_open_r = self._smooth("eye_r", 1.0, dt, 0.25)
            p.viseme = "fechada"
            return p

        h = frame.head
        bs = frame.blendshapes

        ry, rp, rr, rx, ry0 = self._rest or (0.0, 0.0, 0.0, 0.5, 0.5)

        # --- cabeca ---
        p.head_turn = self._smooth("head_turn",
                                   deadzone(clamp((h.yaw - ry) / self.YAW_RANGE)), dt)
        p.head_nod = self._smooth("head_nod",
                                  deadzone(clamp((h.pitch - rp) / self.PITCH_RANGE)), dt)
        p.head_tilt = self._smooth("head_tilt",
                                   deadzone(clamp((h.roll - rr) / self.ROLL_RANGE)), dt)
        p.head_x = self._smooth("head_x", deadzone(clamp((h.x - rx) * 4)), dt, 0.10)
        p.head_y = self._smooth("head_y", deadzone(clamp((h.y - ry0) * 4)), dt, 0.10)

        # --- olhos --- (blendshape e "quanto fechou"; invertemos)
        blink_l = bs.get("eyeBlinkLeft", 0.0)
        blink_r = bs.get("eyeBlinkRight", 0.0)
        # piscada precisa ser rapida: meia-vida curta
        p.eye_open_l = self._smooth("eye_l", 1.0 - remap(blink_l, 0.15, 0.55), dt, 0.025)
        p.eye_open_r = self._smooth("eye_r", 1.0 - remap(blink_r, 0.15, 0.55), dt, 0.025)

        # --- olhar ---
        gx = (bs.get("eyeLookOutLeft", 0) + bs.get("eyeLookInRight", 0)) / 2 \
             - (bs.get("eyeLookInLeft", 0) + bs.get("eyeLookOutRight", 0)) / 2
        gy = (bs.get("eyeLookUpLeft", 0) + bs.get("eyeLookUpRight", 0)) / 2 \
             - (bs.get("eyeLookDownLeft", 0) + bs.get("eyeLookDownRight", 0)) / 2
        p.gaze_x = self._smooth("gaze_x", deadzone(clamp(gx * 2.2), 0.08), dt, 0.05)
        p.gaze_y = self._smooth("gaze_y", deadzone(clamp(gy * 2.2), 0.08), dt, 0.05)

        # --- sobrancelhas --- (levantada positiva, franzida negativa)
        up_l = bs.get("browOuterUpLeft", 0.0)
        up_r = bs.get("browOuterUpRight", 0.0)
        down = bs.get("browDownLeft", 0.0) + bs.get("browDownRight", 0.0)
        inner = bs.get("browInnerUp", 0.0)
        p.brow_l = self._smooth("brow_l", clamp(max(up_l, inner) * 1.6 - down * 0.8), dt, 0.07)
        p.brow_r = self._smooth("brow_r", clamp(max(up_r, inner) * 1.6 - down * 0.8), dt, 0.07)

        # --- boca ---
        jaw = bs.get("jawOpen", 0.0)
        funnel = bs.get("mouthFunnel", 0.0)
        pucker = bs.get("mouthPucker", 0.0)
        smile = (bs.get("mouthSmileLeft", 0) + bs.get("mouthSmileRight", 0)) / 2
        stretch = (bs.get("mouthStretchLeft", 0) + bs.get("mouthStretchRight", 0)) / 2
        # Piso baixo de proposito: com 0.05 o sistema so reagia a boca bem
        # aberta, e falar normal (que mal passa de 0.1 no jawOpen) nao movia
        # nada. A zona morta contra tremor fica na propria curva, nao no piso.
        p.mouth_open = self._smooth("mouth_open", remap(jaw, 0.015, 0.42), dt, 0.030)
        p.mouth_wide = self._smooth("mouth_wide", clamp(max(smile, stretch) * 1.5, 0, 1), dt, 0.06)
        # sorriso e canal proprio: `stretch` tambem alarga a boca, mas ao
        # falar, e nao ao sorrir - misturar os dois fazia o avatar sorrir no
        # meio de uma frase.
        p.smile = self._smooth("smile", clamp(smile * 1.8, 0, 1), dt, 0.08)
        p.viseme = self._viseme(p.mouth_open, p.mouth_wide, pucker, funnel,
                                p.smile)

        return p

    @staticmethod
    def _viseme(open_, wide, pucker, funnel, sorriso):
        # O sorriso vem antes dos visemas de fala: sorrir de boca fechada nao
        # abre o maxilar, entao o canal de abertura fica em zero e, sem este
        # caso, o rosto ficava sempre serio por mais que a pessoa sorrisse.
        if open_ < 0.05:
            return "sorriso" if sorriso > 0.35 else "fechada"
        if sorriso > 0.55 and wide > 0.45:
            return "sorriso_aberto"
        if pucker > 0.35 or funnel > 0.35:
            return "u" if open_ < 0.45 else "o"
        if wide > 0.45:
            return "i" if open_ < 0.45 else "e"
        return "a"
