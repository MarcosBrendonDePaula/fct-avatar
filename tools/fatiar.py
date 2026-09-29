"""Fatia a arte gerada em camadas registradas.

Roda o FaceLandmarker na propria arte para descobrir onde estao olhos, boca
e queixo: assim os recortes saem alinhados sem ninguem medir pixel na mao, e
as variantes (olhos fechados, bocas) recortam a MESMA caixa da base, o que
garante registro perfeito na hora de trocar o sprite.

Uso:  .venv\\Scripts\\python.exe tools/fatiar.py
Saida: art/layers/*.png + art/avatar.json
"""

import json
import sys
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision
from PIL import Image, ImageChops, ImageDraw, ImageFilter

RAIZ = Path(__file__).resolve().parent.parent
ART = RAIZ / "art"
OUT = ART / "layers"

# indices de landmark do FaceLandmarker (canonical face mesh)
OLHO_E = [33, 133, 159, 145, 160, 144, 158, 153]
OLHO_D = [362, 263, 386, 374, 385, 380, 387, 373]
SOBR_E = [70, 63, 105, 66, 107]
SOBR_D = [300, 293, 334, 296, 336]
BOCA = [61, 291, 0, 17, 13, 14, 78, 308, 82, 87]
QUEIXO = 152
ROSTO_OVAL = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397,
              365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58,
              132, 93, 234, 127, 162, 21, 54, 103, 67, 109]



# Pontos que NAO mudam quando o personagem abre a boca ou fecha os olhos:
# ponte do nariz, cantos dos olhos, temporas e testa. Servem de referencia
# para alinhar cada variante a base.
ESTAVEIS = [6, 168, 197, 195, 4, 1, 33, 133, 362, 263, 234, 454, 10, 151]


def alinhar(img, pts, pts_base, size):
    """Encaixa uma variante na base por semelhanca (rotacao+escala+translacao).

    Cada imagem que o modelo devolve vem com um deslocamento global proprio.
    Sem isso, o sprite da boca recortado da variante nao cai no mesmo lugar
    do rosto da base e o recorte aparece torto.
    """
    origem = np.float32([pts[i] for i in ESTAVEIS])
    destino = np.float32([pts_base[i] for i in ESTAVEIS])
    m, _ = cv2.estimateAffinePartial2D(origem, destino, method=cv2.LMEDS)
    if m is None:
        return img
    arr = cv2.warpAffine(np.asarray(img), m, size, flags=cv2.INTER_LANCZOS4,
                         borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))
    desloc = float(np.hypot(m[0, 2], m[1, 2]))
    if desloc > 1.0:
        print(f"  alinhada: deslocamento de {desloc:.1f} px corrigido")
    return Image.fromarray(arr, "RGBA")


def alinhar_por_corpo(img, img_base, pts_base, size):
    """Encaixa uma vista (3/4, cima, baixo) na base pelo CORPO, nao pelo rosto.

    Para as variantes de boca e olho da para alinhar pelos landmarks faciais,
    porque o rosto continua na mesma pose. Aqui nao: o rosto mudou de
    proposito, e alinhar por ele desfaria justamente a virada. O que tem de
    coincidir entre as vistas e o tronco - o prompt pediu ombros parados -,
    entao a referencia e a faixa abaixo da cabeca, casada por correlacao de
    fase.
    """
    _, cy, _, ry = elipse_cabeca(pts_base)
    topo = int(min(size[1] - 2, cy + ry * 1.05))
    if topo >= size[1] - 8:
        return img

    def faixa(im):
        g = cv2.cvtColor(np.asarray(im.convert("RGB")), cv2.COLOR_RGB2GRAY)
        return np.float32(g[topo:, :])

    a, b = faixa(img_base), faixa(img)
    janela = cv2.createHanningWindow((a.shape[1], a.shape[0]), cv2.CV_32F)
    (dx, dy), _ = cv2.phaseCorrelate(a, b, janela)

    if abs(dx) < 0.5 and abs(dy) < 0.5:
        return img
    print(f"  corpo alinhado: {dx:+.1f}, {dy:+.1f} px")
    m = np.float32([[1, 0, -dx], [0, 1, -dy]])
    arr = cv2.warpAffine(np.asarray(img), m, size, flags=cv2.INTER_LANCZOS4,
                         borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))
    return Image.fromarray(arr, "RGBA")


def suavizar_borda(img, faixa=10):
    """Degrada o alfa nas bordas do recorte para o retangulo nao aparecer."""
    w, h = img.size
    rampa = Image.new("L", (w, h), 255)
    d = ImageDraw.Draw(rampa)
    for i in range(faixa):
        v = int(255 * (i + 1) / (faixa + 1))
        d.rectangle([i, i, w - 1 - i, h - 1 - i], outline=v)
    img.putalpha(ImageChops.multiply(img.split()[3], rampa))
    return img


def landmarks(caminho):
    opts = vision.FaceLandmarkerOptions(
        base_options=mp_python.BaseOptions(
            model_asset_path=str(RAIZ / "models" / "face_landmarker.task")
        ),
        running_mode=vision.RunningMode.IMAGE,
        num_faces=1,
    )
    with vision.FaceLandmarker.create_from_options(opts) as det:
        img = Image.open(caminho).convert("RGBA")
        # o modelo nao entende alfa: compoe sobre cinza medio antes de detectar
        fundo = Image.new("RGBA", img.size, (128, 128, 128, 255))
        fundo.alpha_composite(img)
        rgb = np.ascontiguousarray(np.asarray(fundo.convert("RGB")))
        res = det.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))
    if not res.face_landmarks:
        raise SystemExit(f"Nenhum rosto encontrado em {caminho}")
    w, h = img.size
    return [(p.x * w, p.y * h) for p in res.face_landmarks[0]], img.size


def caixa(pts, indices, margem_x, margem_y, tamanho):
    xs = [pts[i][0] for i in indices]
    ys = [pts[i][1] for i in indices]
    x0 = max(0, min(xs) - margem_x)
    x1 = min(tamanho[0], max(xs) + margem_x)
    y0 = max(0, min(ys) - margem_y)
    y1 = min(tamanho[1], max(ys) + margem_y)
    return tuple(int(v) for v in (x0, y0, x1, y1))


def elipse_cabeca(pts):
    oval = [pts[i] for i in ROSTO_OVAL]
    cx = sum(x for x, _ in oval) / len(oval)
    topo = min(y for _, y in oval)
    base_y = max(y for _, y in oval)
    largura = max(x for x, _ in oval) - min(x for x, _ in oval)
    rx = largura * 0.95          # o cabelo passa bem do oval do rosto
    ry = (base_y - topo) * 0.82
    return cx, (topo + base_y) / 2 - ry * 0.12, rx, ry


def furar_corpo(img, pts, size, encolher=0.86):
    """Apaga a cabeca da camada do corpo e preenche o buraco.

    O corpo era a arte inteira, cabeca incluida. Bastava mexer a cabeca para
    o rosto original aparecer por tras dela - dois rostos na tela.

    O furo e um pouco MENOR que a elipse da camada da cabeca, para que a
    cabeca continue cobrindo o buraco mesmo deslocada. O que sobra e
    preenchido por inpaint: fica borrado, mas so aparece de relance na
    fresta, e borrado e muito melhor que um segundo rosto.
    """
    cx, cy, rx, ry = elipse_cabeca(pts)
    buraco = Image.new("L", size, 0)
    ImageDraw.Draw(buraco).ellipse(
        [cx - rx * encolher, cy - ry * encolher,
         cx + rx * encolher, cy + ry * encolher + ry * 0.30], fill=255
    )
    mascara = np.asarray(buraco)
    arr = np.asarray(img)
    rgb = cv2.inpaint(cv2.cvtColor(arr, cv2.COLOR_RGBA2BGR), mascara, 12,
                      cv2.INPAINT_TELEA)

    # Escurece o preenchimento: o borrado continua la, mas quando aparece na
    # fresta ele le como sombra atras da cabeca, que e o que deveria haver
    # ali mesmo - e nao como um borrao.
    sombra = (rgb.astype(np.float32) * 0.45).astype(np.uint8)
    m3 = cv2.cvtColor(cv2.GaussianBlur(mascara, (31, 31), 0), cv2.COLOR_GRAY2BGR)
    rgb = cv2.add(cv2.multiply(sombra, m3, scale=1 / 255.0),
                  cv2.multiply(rgb, cv2.bitwise_not(m3), scale=1 / 255.0))

    out = np.dstack([cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB), arr[:, :, 3]])
    return Image.fromarray(out, "RGBA")


def recortar_cabeca(img, pts, size, destino, extra_baixo=0.35):
    """Salva a arte recortada por uma elipse suave em volta do rosto e cabelo.

    A elipse vem dos landmarks, nao de numeros fixos, para funcionar igual na
    vista frontal e nas de 3/4.
    """
    oval = [pts[i] for i in ROSTO_OVAL]
    cx = sum(x for x, _ in oval) / len(oval)
    topo = min(y for _, y in oval)
    base_y = max(y for _, y in oval)
    largura = max(x for x, _ in oval) - min(x for x, _ in oval)
    rx = largura * 0.95          # o cabelo passa bem do oval do rosto
    ry = (base_y - topo) * 0.82
    cy = (topo + base_y) / 2 - ry * 0.12

    mascara = Image.new("L", size, 0)
    ImageDraw.Draw(mascara).ellipse(
        [cx - rx, cy - ry, cx + rx, cy + ry + ry * extra_baixo], fill=255
    )
    mascara = mascara.filter(ImageFilter.GaussianBlur(14))

    cabeca = img.copy()
    cabeca.putalpha(ImageChops.multiply(cabeca.split()[3], mascara))
    cabeca.save(destino)
    return cx, cy, rx, ry


def recortar(origem, box, destino):
    Image.open(origem).convert("RGBA").crop(box).save(destino)


def main():
    base = ART / "base.png"
    if not base.exists():
        raise SystemExit(f"Falta a arte base em {base}")
    pts, size = landmarks(base)
    OUT.mkdir(parents=True, exist_ok=True)

    # margens generosas: o olho aberto tem cilios e cantos que passam bem do
    # contorno dos landmarks. Caixa apertada deixa o olho antigo aparecendo
    # por baixo do sprite novo.
    b_olho_e = caixa(pts, OLHO_E, 58, 52, size)
    b_olho_d = caixa(pts, OLHO_D, 58, 52, size)
    b_sobr_e = caixa(pts, SOBR_E, 22, 20, size)
    b_sobr_d = caixa(pts, SOBR_D, 22, 20, size)
    b_boca = caixa(pts, BOCA, 72, 64, size)
    queixo_y = int(pts[QUEIXO][1])

    manifesto = {
        "base": "base.png",
        "size": list(size),
        "corte_pescoco": queixo_y + 30,
        "caixas": {
            "olho_e": b_olho_e, "olho_d": b_olho_d,
            "sobr_e": b_sobr_e, "sobr_d": b_sobr_d,
            "boca": b_boca,
        },
        "camadas": {},
    }

    img_base = Image.open(base).convert("RGBA")

    # Corpo e a arte inteira; a cabeca e a MESMA arte recortada por uma elipse
    # de borda suave em volta do rosto e do cabelo.
    #
    # Um corte horizontal levava junto o topo do capuz, e qualquer movimento da
    # cabeca deixava uma emenda reta visivel no ombro. Com a elipse, o que se
    # move e so a cabeca, e o corpo intacto por baixo preenche o resto - a
    # borda difusa esconde o encontro das duas camadas.
    corpo = furar_corpo(img_base, pts, size)
    corpo.save(OUT / "corpo.png")
    manifesto["camadas"]["corpo"] = {"arquivo": "corpo.png", "pos": [0, 0]}

    cx, cy, rx, ry = recortar_cabeca(img_base, pts, size, OUT / "cabeca.png")
    manifesto["camadas"]["cabeca"] = {"arquivo": "cabeca.png", "pos": [0, 0]}
    manifesto["elipse_cabeca"] = [cx, cy, rx, ry]

    # vistas de 3/4: cada uma calcula a propria elipse pelos landmarks dela,
    # entao o recorte acompanha a cabeca virada sem ninguem ajustar na mao
    # a vista de cima expoe muito mais pescoco, entao a elipse precisa descer
    # mais para continuar cobrindo o furo aberto na camada do corpo
    for nome, arquivo, extra in (("cabeca_esq", "vira_esq.png", 0.35),
                                 ("cabeca_dir", "vira_dir.png", 0.35),
                                 ("cabeca_cima", "olha_cima.png", 0.85),
                                 ("cabeca_baixo", "olha_baixo.png", 0.45)):
        origem = ART / arquivo
        if not origem.exists():
            continue
        print(f"{nome}:")
        img_v = alinhar_por_corpo(Image.open(origem).convert("RGBA"),
                                  img_base, pts, size)
        tmp = ART / f".alinhada_{arquivo}"
        img_v.save(tmp)
        pts_v, size_v = landmarks(tmp)
        tmp.unlink()
        recortar_cabeca(img_v, pts_v, size_v, OUT / f"{nome}.png", extra)
        manifesto["camadas"][nome] = {"arquivo": f"{nome}.png", "pos": [0, 0]}

    # sprites de troca: mesma caixa, arquivos diferentes
    variantes = {
        "olhos_abertos": (base, [("olho_e", b_olho_e), ("olho_d", b_olho_d)]),
        "olhos_fechados": (ART / "olhos_fechados.png",
                           [("olho_e", b_olho_e), ("olho_d", b_olho_d)]),
        "boca_fechada": (base, [("boca", b_boca)]),
        "boca_a": (ART / "boca_a.png", [("boca", b_boca)]),
        "boca_i": (ART / "boca_i.png", [("boca", b_boca)]),
        "boca_o": (ART / "boca_o.png", [("boca", b_boca)]),
    }

    faltando = []
    for nome, (origem, pecas) in variantes.items():
        if not Path(origem).exists():
            faltando.append(nome)
            continue
        img_v = Image.open(origem).convert("RGBA")
        if Path(origem) != base:
            print(f"{nome}:")
            pts_v, _ = landmarks(origem)
            img_v = alinhar(img_v, pts_v, pts, size)
        for peca, box in pecas:
            arq = f"{nome}__{peca}.png"
            suavizar_borda(img_v.crop(box)).save(OUT / arq)
            manifesto["camadas"][f"{nome}:{peca}"] = {
                "arquivo": arq, "pos": [box[0], box[1]]
            }

    (ART / "avatar.json").write_text(json.dumps(manifesto, indent=2), encoding="utf-8")
    print(f"elipse da cabeca: centro=({cx:.0f},{cy:.0f}) raios=({rx:.0f},{ry:.0f})")
    print(f"caixas: olho_e={b_olho_e} boca={b_boca}")
    print(f"{len(manifesto['camadas'])} camadas escritas em {OUT}")
    if faltando:
        print(f"variantes ainda nao geradas: {', '.join(faltando)}", file=sys.stderr)


if __name__ == "__main__":
    main()
