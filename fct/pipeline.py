"""Tracking numa thread propria, desacoplado do desenho.

Com tudo no mesmo laco, cada quadro exibido esperava o MediaPipe terminar -
o desenho andava no passo do tracking e a imagem engasgava. Aqui a thread
publica sempre o ultimo AvatarFrame e o laco de render consome quando
quiser: como a suavizacao em state.py interpola pelo relogio, desenhar a
60 com tracking a 25 sai fluido, nao "travado a 25".
"""

import threading
import time

from .capture import Capture
from .tracker import Tracker
from .types import AvatarFrame


class TrackingThread:
    def __init__(self, camera=0, track_hands=True, ajuste=None):
        self.ajuste = ajuste
        self._cap = Capture(camera)
        self._tracker = Tracker(track_hands=track_hands)
        self._frame = AvatarFrame(timestamp=time.time())
        self._cam_frame = None
        self._lock = threading.Lock()
        self._parar = threading.Event()
        self.fps = 0.0                # ritmo do tracking, para o painel
        self._t = threading.Thread(target=self._loop, daemon=True)
        self._t.start()

    def _loop(self):
        ultimo_id, t_prev = -1, time.time()
        while not self._parar.is_set():
            if self._cap.frame_id == ultimo_id:
                time.sleep(0.001)     # nada novo: nao queima CPU a toa
                continue
            ultimo_id = self._cap.frame_id
            frame = self._cap.read()
            if frame is None:
                continue
            if self.ajuste is not None:
                frame = self.ajuste.aplicar(frame)

            av = self._tracker.process(frame)
            agora = time.time()
            self.fps = 0.9 * self.fps + 0.1 / max(1e-6, agora - t_prev)
            t_prev = agora
            with self._lock:
                self._frame = av
                self._cam_frame = frame

    def ultimo(self):
        """(AvatarFrame, frame da camera) mais recentes."""
        with self._lock:
            return self._frame, self._cam_frame

    def close(self):
        self._parar.set()
        self._t.join(timeout=1.5)
        self._tracker.close()
        self._cap.release()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
