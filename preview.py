"""Preview do avatar em janela (sem webcam virtual).

Uso:  .venv\\Scripts\\python.exe preview.py
Teclas:  C = calibrar repouso    D = debug on/off    Q/ESC = sair
"""

import argparse
import time

import cv2
import numpy as np

from fct.capture import Capture
from fct.renderer import Renderer
from fct.state import StateMapper
from fct.tracker import Tracker

FUNDO = (26, 28, 34)


def rgba_para_bgr(img, fundo=FUNDO):
    arr = np.asarray(img).astype(np.float32)
    rgb, a = arr[:, :, :3], arr[:, :, 3:4] / 255.0
    base = np.array(fundo[::-1], dtype=np.float32)
    out = rgb * a + base * (1 - a)
    return cv2.cvtColor(out.astype(np.uint8), cv2.COLOR_RGB2BGR)


def painel_debug(frame_cam, params, fps):
    h = 720
    cam = cv2.resize(frame_cam, (int(frame_cam.shape[1] * (h / 2) / frame_cam.shape[0]),
                                 h // 2))
    info = np.full((h - cam.shape[0], cam.shape[1], 3), 18, dtype=np.uint8)
    linhas = [
        f"fps        {fps:5.1f}",
        f"rosto      {'sim' if params.present else 'NAO'}",
        f"turn/nod   {params.head_turn:+.2f} {params.head_nod:+.2f}",
        f"tilt       {params.head_tilt:+.2f}",
        f"pos x/y    {params.head_x:+.2f} {params.head_y:+.2f}",
        f"olhos      {params.eye_open_l:.2f} {params.eye_open_r:.2f}",
        f"olhar      {params.gaze_x:+.2f} {params.gaze_y:+.2f}",
        f"sobranc.   {params.brow_l:+.2f} {params.brow_r:+.2f}",
        f"boca       {params.mouth_open:.2f} w{params.mouth_wide:.2f}",
        f"viseme     {params.viseme}",
    ]
    for i, t in enumerate(linhas):
        cv2.putText(info, t, (14, 30 + i * 28), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (180, 220, 180), 1, cv2.LINE_AA)
    return np.vstack([cam, info])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--no-hands", action="store_true")
    ap.add_argument("--debug", action="store_true", default=True)
    args = ap.parse_args()

    tracker = Tracker(track_hands=not args.no_hands)
    mapper = StateMapper()
    renderer = Renderer()
    debug = args.debug
    fps, t_prev = 0.0, time.time()

    print("Preview rodando. C = calibrar, D = debug, Q/ESC = sair.")
    with Capture(args.camera) as cap:
        ultimo_id, av = -1, None
        while True:
            frame = cap.read()
            if frame is None:
                continue

            # so roda o tracking em frame novo; o render continua em toda volta
            # pra que a animacao idle nao engasgue quando a camera atrasa.
            if cap.frame_id != ultimo_id:
                ultimo_id = cap.frame_id
                av = tracker.process(frame)
            params = mapper.update(av)
            img = rgba_para_bgr(renderer.render(params))

            agora = time.time()
            fps = 0.9 * fps + 0.1 / max(1e-6, agora - t_prev)
            t_prev = agora

            if debug:
                img = np.hstack([img, painel_debug(frame, params, fps)])

            cv2.imshow("Avatar - preview", img)
            k = cv2.waitKey(1) & 0xFF
            if k in (27, ord("q")):
                break
            if k == ord("c"):
                mapper.calibrate(av)
                print("repouso calibrado")
            if k == ord("d"):
                debug = not debug
                cv2.destroyAllWindows()

    tracker.close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
