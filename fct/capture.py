"""Captura de webcam. Nao sabe nada sobre rostos."""

import threading

import cv2


class Capture:
    def __init__(self, index=0, width=1280, height=720, fps=30):
        self.cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
        if not self.cap.isOpened():
            raise RuntimeError(
                f"Nao consegui abrir a webcam {index}. "
                "Feche outros apps que possam estar usando a camera."
            )
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.cap.set(cv2.CAP_PROP_FPS, fps)
        try:
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except cv2.error:
            pass

        # A leitura bloqueia ate a camera entregar o proximo frame. Numa thread,
        # essa espera sobrepoe com o tracking em vez de somar a ele; o laco
        # principal sempre pega o frame mais recente, nunca uma fila atrasada.
        self._frame = None
        self.frame_id = 0
        self._lock = threading.Lock()
        self._parar = threading.Event()
        self._pronto = threading.Event()
        self._t = threading.Thread(target=self._loop, daemon=True)
        self._t.start()
        if not self._pronto.wait(timeout=5.0):
            self.release()
            raise RuntimeError(
                f"A webcam {index} abriu mas nao entregou nenhum frame em 5s."
            )

    def _loop(self):
        while not self._parar.is_set():
            ok, frame = self.cap.read()
            if not ok:
                continue
            frame = cv2.flip(frame, 1)  # espelho: mexo pra direita, avatar vai pra direita
            with self._lock:
                self._frame = frame
                self.frame_id += 1
            self._pronto.set()

    def read(self):
        """Frame BGR mais recente, ou None enquanto o primeiro nao chegou."""
        with self._lock:
            return None if self._frame is None else self._frame.copy()

    def release(self):
        self._parar.set()
        if hasattr(self, '_t'):
            self._t.join(timeout=1.0)
        self.cap.release()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.release()
