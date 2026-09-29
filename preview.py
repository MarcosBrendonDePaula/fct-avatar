"""Preview do avatar em janela (sem webcam virtual).

Uso:  .venv\\Scripts\\python.exe preview.py

Teclas:
  C          calibra a pose de repouso (fique neutro e aperte)
  A          liga/desliga o ajuste automatico de brilho
  + / -      ganho (clareia/escurece) quando em manual
  [ / ]      gama (levanta/abaixa as sombras)
  M          mostra/esconde a malha do rosto sobre a camera
  D          mostra/esconde o painel de debug
  P          alterna entre a arte e o boneco placeholder
  Q / ESC    sai
"""

import argparse
import time

import cv2
import numpy as np

from fct.pipeline import TrackingThread
from fct.imagem import Ajuste
from fct.renderer import Renderer
from fct.sprite_renderer import SpriteRenderer
from fct.state import StateMapper

FUNDO = (26, 28, 34)
VERDE = (150, 240, 160)
AMARELO = (120, 220, 250)
VERMELHO = (110, 110, 250)

# grupos de landmarks desenhados na malha, para dar pra ver o que o tracker ve
GRUPOS = {
    "olho_e": [33, 160, 158, 133, 153, 144],
    "olho_d": [362, 385, 387, 263, 373, 380],
    "boca": [61, 39, 0, 269, 291, 405, 17, 181],
    "sobr_e": [70, 63, 105, 66, 107],
    "sobr_d": [300, 293, 334, 296, 336],
}


def rgba_para_bgr(img):
    """O renderer do preview ja desenha sobre um fundo opaco, entao aqui e so
    trocar a ordem dos canais. Compor em float32 custava 50ms por frame."""
    return cv2.cvtColor(np.asarray(img), cv2.COLOR_RGBA2BGR)


def desenhar_malha(cam, av):
    """Pontos do rosto e das maos sobre a imagem da camera."""
    if not av or not av.face_points:
        return cam
    h, w = cam.shape[:2]
    pts = [(int(x * w), int(y * h)) for x, y in av.face_points]

    # nuvem completa, discreta
    for x, y in pts[::4]:
        cv2.circle(cam, (x, y), 1, (90, 160, 110), -1, cv2.LINE_AA)
    # contornos que realmente controlam o avatar, destacados
    for nome, idxs in GRUPOS.items():
        cor = AMARELO if nome.startswith("olho") else VERDE
        poly = np.array([pts[i] for i in idxs if i < len(pts)], np.int32)
        cv2.polylines(cam, [poly], True, cor, 1, cv2.LINE_AA)

    for mao in av.hands:
        for lx, ly, _ in mao.landmarks:
            cv2.circle(cam, (int(lx * w), int(ly * h)), 2, (240, 180, 90), -1,
                       cv2.LINE_AA)
    return cam


def painel(cam, params, ajuste, fps, fps_track, malha, av):
    alt = 720
    largura = int(cam.shape[1] * (alt / 2) / cam.shape[0])
    cam = cv2.resize(cam, (largura, alt // 2))
    if malha:
        cam = desenhar_malha(cam, av)

    info = np.full((alt - cam.shape[0], largura, 3), 18, dtype=np.uint8)
    if params.present:
        estado, cor_estado = "RASTREANDO", VERDE
    else:
        estado, cor_estado = "SEM ROSTO", VERMELHO

    if ajuste.brilho < 45:
        aviso, cor_aviso = "escuro demais - aperte A ou +", VERMELHO
    elif ajuste.brilho > 210:
        aviso, cor_aviso = "estourado - aperte -", VERMELHO
    else:
        aviso, cor_aviso = "iluminacao ok", VERDE

    linhas = [
        (f"{estado}", cor_estado),
        (f"desenho {fps:.0f} fps   tracking {fps_track:.0f} fps"
         f"   brilho {ajuste.brilho:.0f}", (190, 190, 190)),
        (f"{ajuste.resumo()}", (190, 190, 190)),
        (aviso, cor_aviso),
        ("", None),
        (f"turn {params.head_turn:+.2f}  nod {params.head_nod:+.2f}"
         f"  tilt {params.head_tilt:+.2f}", (200, 200, 200)),
        (f"pos  {params.head_x:+.2f} {params.head_y:+.2f}", (200, 200, 200)),
        (f"olhos {params.eye_open_l:.2f} {params.eye_open_r:.2f}"
         f"   olhar {params.gaze_x:+.2f} {params.gaze_y:+.2f}", (200, 200, 200)),
        (f"sobrancelhas {params.brow_l:+.2f} {params.brow_r:+.2f}", (200, 200, 200)),
        (f"boca {params.mouth_open:.2f}  largura {params.mouth_wide:.2f}"
         f"  viseme {params.viseme}", (200, 200, 200)),
        (f"maos {len(av.hands) if av else 0}", (200, 200, 200)),
        ("", None),
        ("C calibrar | A auto | +- ganho | [] gama | M malha | P arte", (120, 120, 120)),
    ]
    y = 26
    for texto, cor in linhas:
        if texto:
            cv2.putText(info, texto, (14, y), cv2.FONT_HERSHEY_SIMPLEX, 0.52,
                        cor, 1, cv2.LINE_AA)
        y += 24
    return np.vstack([cam, info])


def criar_renderer(placeholder, bg):
    if placeholder:
        return Renderer(bg=bg)
    try:
        return SpriteRenderer(bg=bg)
    except (FileNotFoundError, RuntimeError) as e:
        print(f"{e}\nCaindo no boneco placeholder.")
        return Renderer(bg=bg)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--no-hands", action="store_true")
    ap.add_argument("--placeholder", action="store_true",
                    help="usa o boneco desenhado em codigo em vez da arte")
    ap.add_argument("--ganho", type=float, default=1.0)
    ap.add_argument("--gama", type=float, default=1.0)
    ap.add_argument("--fps", type=float, default=45,
                    help="limite do laco de desenho; desenhar solto rouba CPU "
                         "da thread de tracking e o movimento fica pior")
    ap.add_argument("--auto-brilho", action="store_true",
                    help="ajusta o ganho sozinho (bom para quarto escuro)")
    args = ap.parse_args()

    mapper = StateMapper()
    ajuste = Ajuste(ganho=args.ganho, gama=args.gama, auto=args.auto_brilho)
    fundo = FUNDO[::-1] + (255,)   # opaco: converter fica barato
    placeholder = args.placeholder
    renderer = criar_renderer(placeholder, fundo)

    debug, malha = True, True
    fps, t_prev = 0.0, time.time()
    print(__doc__)

    with TrackingThread(args.camera, track_hands=not args.no_hands,
                        ajuste=ajuste) as pipe:
        prox = time.time()
        while True:
            agora = time.time()
            if agora < prox:
                time.sleep(min(0.003, prox - agora))
                continue
            prox = agora + 1.0 / args.fps

            av, frame = pipe.ultimo()
            if frame is None:
                continue
            params = mapper.update(av)
            img = rgba_para_bgr(renderer.render(params))

            agora = time.time()
            fps = 0.9 * fps + 0.1 / max(1e-6, agora - t_prev)
            t_prev = agora

            if debug:
                img = np.hstack([img, painel(frame, params, ajuste, fps,
                                             pipe.fps, malha, av)])
            cv2.imshow("Avatar - preview", img)

            k = cv2.waitKey(1) & 0xFF
            if k in (27, ord("q")):
                break
            elif k == ord("c"):
                mapper.calibrate(av)
                print("repouso calibrado")
            elif k == ord("a"):
                ajuste.auto = not ajuste.auto
                print(f"brilho automatico: {'on' if ajuste.auto else 'off'}")
            elif k in (ord("+"), ord("=")):
                ajuste.auto = False
                ajuste.ganho = min(4.0, ajuste.ganho + 0.1)
            elif k in (ord("-"), ord("_")):
                ajuste.auto = False
                ajuste.ganho = max(0.2, ajuste.ganho - 0.1)
            elif k == ord("]"):
                ajuste.gama = min(3.0, ajuste.gama + 0.1)
            elif k == ord("["):
                ajuste.gama = max(0.3, ajuste.gama - 0.1)
            elif k == ord("m"):
                malha = not malha
            elif k == ord("p"):
                placeholder = not placeholder
                renderer = criar_renderer(placeholder, fundo)
            elif k == ord("d"):
                debug = not debug
                cv2.destroyAllWindows()

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
