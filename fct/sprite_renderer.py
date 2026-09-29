"""Renderer baseado nos PNGs recortados de art/.

Interface: render(AvatarParams) -> imagem BGR (numpy), pronta para exibir.

Tudo acontece em numpy/cv2, em BGR. A versao anterior ia e voltava entre
PIL e numpy a cada frame e compunha camadas do tamanho do quadro inteiro;
custava 10,5 ms e limitava o desenho a ~47 fps. As tres mudancas que
importam, todas medidas:

1. O corpo e o fundo sao compostos UMA vez, no construtor. Por frame so
   sobra copiar esse plano pronto.
2. As camadas ficam recortadas na caixa util (a cabeca ocupa metade do
   quadro), entao a rotacao e a mistura tocam bem menos pixels.
3. Nada de PIL no caminho quente: converter RGBA->BGR custava 2,6 ms por
   frame sozinho e agora nem existe, porque as camadas ja sao carregadas
   em BGR.
"""

import json
import time
from pathlib import Path

import cv2
import numpy as np
from .camada import Camada

RAIZ = Path(__file__).resolve().parent.parent
ART = RAIZ / "art"

GIRO_MAX = 0.16       # radianos de giro no cilindro em head_turn = 1
ACENO_MAX = 0.13      # idem para o acenar
SOBRANCELHA_MAX = 14   # px que a sobrancelha sobe em brow = 1
BALANCO_CABELO = 34    # px de balanco do cabelo em atraso maximo
LIMITE_CILINDRO = 0.50  # passando disso, usa a arte desenhada naquela pose
MARGEM_GIRO = 90      # folga em volta da cabeca para a rotacao nao cortar
MARGEM_CORPO = 40     # idem para o tronco

# viseme -> sprite disponivel (e/u reaproveitam os vizinhos mais proximos)
VISEME_SPRITE = {
    "fechada": "boca_fechada", "a": "boca_a", "i": "boca_i",
    "e": "boca_i", "o": "boca_o", "u": "boca_o",
    "sorriso": "sorriso", "sorriso_aberto": "sorriso_aberto",
}


class Mola:
    """Mola amortecida. Persegue o alvo com atraso e passa um pouco dele.

    E o que da o balanco do cabelo: o alvo e o movimento da cabeca, e o que
    interessa e a DIFERENCA entre a mola e o alvo - o quanto o cabelo ficou
    para tras. Sem a ultrapassagem o cabelo so arrastaria, sem devolver.
    """

    def __init__(self, k=120.0, amortecimento=13.0):
        self.k = k
        self.c = amortecimento
        self.valor = 0.0
        self.vel = 0.0

    def atualizar(self, alvo, dt):
        dt = min(dt, 0.05)            # passo grande faz a mola explodir
        acel = self.k * (alvo - self.valor) - self.c * self.vel
        self.vel += acel * dt
        self.valor += self.vel * dt
        return self.valor


def col_bruta(cx, r, seno, yaw):
    return cx + r * np.sin(np.arcsin(seno) - yaw)


class SpriteRenderer:
    def __init__(self, saida=(720, 720), bg=(0, 0, 0, 0), manifesto=None):
        caminho = Path(manifesto or ART / "avatar.json")
        if not caminho.exists():
            raise FileNotFoundError(
                f"Falta {caminho}. Rode:  .venv\\Scripts\\python.exe tools/fatiar.py"
            )
        self.m = json.loads(caminho.read_text(encoding="utf-8"))
        self.size = saida
        self.escala = saida[1] / self.m["size"][1]
        self._vista = None
        self._faltando = set()
        self._peso_cab = None
        self._mola = Mola()
        self._t_anterior = None

        base_dir = ART / "layers"
        cam = {}
        for nome, info in self.m["camadas"].items():
            arq = base_dir / info["arquivo"]
            if arq.exists():
                cam[nome] = (Camada.de_arquivo(arq, self.escala),
                             (round(info["pos"][0] * self.escala),
                              round(info["pos"][1] * self.escala)))

        w, h = saida
        self.plano = np.empty((h, w, 3), np.uint8)
        self.plano[:] = bg[:3]

        # O corpo e camada animada, recortada na caixa util. Ele era composto
        # uma vez no fundo, o que era barato mas deixava o tronco duro: so a
        # cabeca se mexia, e no pescoco dava para ver.
        corpo = cam.pop("corpo", None)
        self.corpo = None
        if corpo is not None:
            cx0, cy0, cx1, cy1 = corpo[0].caixa_util()
            self.bx0 = max(0, cx0 - MARGEM_CORPO)
            self.by0 = max(0, cy0 - MARGEM_CORPO)
            self.corpo = corpo[0].recortar(
                self.by0, min(h, cy1 + MARGEM_CORPO),
                self.bx0, min(w, cx1 + MARGEM_CORPO))
            # pivo do tronco: embaixo, no centro - o corpo inclina a partir da
            # cintura, nao do meio do peito
            self.pivo_corpo = (self.corpo.forma[1] / 2, self.corpo.forma[0] * 1.15)

        # cabecas recortadas na caixa util, com folga para a rotacao
        self.cabecas = {}
        x0, y0, x1, y1 = cam["cabeca"][0].caixa_util()
        self.hx0 = max(0, x0 - MARGEM_GIRO)
        self.hy0 = max(0, y0 - MARGEM_GIRO)
        hx1 = min(w, x1 + MARGEM_GIRO)
        hy1 = min(h, y1 + MARGEM_GIRO)
        for nome in ("cabeca", "cabeca_esq", "cabeca_dir",
                     "cabeca_cima", "cabeca_baixo"):
            if nome in cam:
                self.cabecas[nome] = cam[nome][0].recortar(
                    self.hy0, hy1, self.hx0, hx1)

        # sprites de olho e boca, em coordenadas da caixa da cabeca
        self.pecas = {}
        for nome, (img, (px, py)) in cam.items():
            if nome.startswith("cabeca"):
                continue
            self.pecas[nome] = (img, (px - self.hx0, py - self.hy0))

        alt, larg = self.cabecas["cabeca"].forma
        gx, gy = np.meshgrid(np.arange(larg, dtype=np.float32),
                             np.arange(alt, dtype=np.float32))
        self._grade = (gx, gy)
        self._buf = (np.empty_like(gx), np.empty_like(gy))

        self.pivo = (self.cabecas["cabeca"].forma[1] / 2,
                     self.m["corte_pescoco"] * self.escala - self.hy0)

    # ------------------------------------------------------------------ util
    def _peca(self, nome):
        p = self.pecas.get(nome)
        if p is None and nome not in self._faltando:
            self._faltando.add(nome)
            print(f"[sprite] camada ausente: {nome}")
        return p

    def _colar(self, alvo, nome, dx=0, dy=0):
        p = self._peca(nome)
        if p is not None:
            alvo.sobrepor(p[0], p[1][0] + int(dx), p[1][1] + int(dy))

    def _mesclar(self, alvo, nome_a, nome_b, t):
        """Cola a mistura de dois sprites da MESMA caixa (t=0 -> a, t=1 -> b)."""
        pa, pb = self._peca(nome_a), self._peca(nome_b)
        if pa is None or pb is None:
            self._colar(alvo, nome_b if pa is None else nome_a)
            return
        alvo.sobrepor(pa[0].misturar(pb[0], t), *pa[1])

    def _peso_cabelo(self):
        """Peso por linha: 1 no alto da cabeca, 0 na altura do pescoco.

        Deixa o balanco agir so onde ha cabelo. Aplicado uniformemente, a
        cara inteira escorregaria junto.
        """
        if self._peso_cab is None:
            alt = self.cabecas["cabeca"].forma[0]
            y = np.arange(alt, dtype=np.float32)
            topo = self.pivo[1] - (self.pivo[1] - MARGEM_GIRO) * 0.95
            t = np.clip((self.pivo[1] - y) / max(1.0, self.pivo[1] - topo), 0, 1)
            self._peso_cab = (t * t * (3 - 2 * t)).astype(np.float32)
        return self._peso_cab

    def _mapa_cilindro(self, yaw, pitch, balanco=0.0):
        """Mapa de remap que gira a cabeca como se fosse um cilindro.

        Tentei antes morfar entre a arte frontal e a de 3/4 por fluxo optico.
        Nao funciona: as duas artes sao geracoes independentes, nao o mesmo
        desenho rotacionado, entao nao ha correspondencia real entre os fios
        de cabelo - e a orelha que surge de um lado nao tem de onde vir.
        Ficou com fantasma e custando 40 ms.

        Projetar a arte frontal num cilindro resolve o problema pela raiz:
        ha uma imagem so, entao nao existe fantasma possivel, e o movimento e
        continuo por construcao. Vale ate uns 25 graus; alem disso a arte de
        3/4 assume.

        O mapa horizontal depende so da coluna, e o vertical so da linha, o
        que deixa a conta barata: dois vetores, nao duas matrizes.
        """
        alt, larg = self.cabecas["cabeca"].forma
        gx, gy = self._grade
        # buffers reaproveitados: alocar dois float32 de 1,7 MB por quadro
        # respondia pela maior parte dos 6,6 ms que este mapa custava
        bx, by = self._buf

        if abs(yaw) > 1e-3:
            cx = larg / 2
            x = np.arange(larg, dtype=np.float32)
            r = larg * 0.62                       # raio do cilindro
            seno = np.clip((x - cx) / r, -1, 1)
            col = self._preservar_borda(col_bruta(cx, r, seno, yaw), x, cx,
                                        larg / 2)
            np.copyto(bx, col)
            mx = bx
        else:
            mx = gx

        if abs(pitch) > 1e-3:
            cy = self.pivo[1] * 0.55
            y = np.arange(alt, dtype=np.float32)
            r = alt * 0.70
            seno = np.clip((y - cy) / r, -1, 1)
            lin = cy + r * np.sin(np.arcsin(seno) - pitch)
            lin = self._preservar_borda(lin, y, cy, alt / 2)
            np.copyto(by, lin[:, None])
            my = by
        else:
            my = gy

        if abs(balanco) > 0.05:
            # balanco do cabelo: deslocamento horizontal que cresce para o
            # alto da cabeca. O sinal e invertido porque o mapa e
            # destino->origem: para o cabelo ir para a direita, buscamos o
            # pixel mais a esquerda.
            if mx is gx:
                np.copyto(bx, gx)
                mx = bx
            np.subtract(mx, balanco * self._peso_cabelo()[:, None], out=mx)

        return mx, my

    @staticmethod
    def _preservar_borda(mapa, eixo, centro, meia, inicio=0.55):
        """Desfaz a deformacao perto da borda, deixando a silhueta parada.

        Deformando ate a borda, o contorno da cabeca encolhia e descobria o
        furo aberto na camada do corpo - aparecia uma mancha escura ao lado
        do cabelo. Prendendo a borda, o rosto gira POR DENTRO de uma cabeca
        de contorno fixo, que e o truque dos rigs 2D.
        """
        d = np.clip((np.abs(eixo - centro) / meia - inicio) / (1 - inicio), 0, 1)
        peso = 1 - d * d * (3 - 2 * d)
        return mapa * peso + eixo * (1 - peso)

    def _decidir_vista(self, p):
        """Vista de arte a usar, ou None para a frontal deformada.

        Ate LIMITE_CILINDRO a cabeca frontal e girada pelo cilindro, que e
        continuo e nunca duplica tracos. Passando disso o cilindro comeca a
        achatar demais, e a arte desenhada naquela pose assume.
        """
        if abs(p.head_turn) > LIMITE_CILINDRO:
            nome = "cabeca_dir" if p.head_turn > 0 else "cabeca_esq"
            if nome in self.cabecas:
                return nome
        if abs(p.head_nod) > LIMITE_CILINDRO:
            nome = "cabeca_cima" if p.head_nod > 0 else "cabeca_baixo"
            if nome in self.cabecas:
                return nome
        return None

    # --------------------------------------------------------------- publico
    def render(self, p):
        # Giro e acenar: PNG 2D nao gira em 3D, entao ha artes em 3/4, de cima
        # e de baixo, e o caminho entre elas e MORFADO pelo fluxo optico. Uma
        # mistura simples sobrepunha dois narizes; o morph leva os tracos de
        # uma pose a outra.
        agora = time.time()
        dt = 1 / 60 if self._t_anterior is None else agora - self._t_anterior
        self._t_anterior = agora
        # o cabelo persegue o movimento lateral da cabeca com atraso; o que
        # balanca e a diferenca entre a mola e o alvo
        alvo = p.head_x * 1.0 + p.head_turn * 0.7 + p.head_tilt * 0.5
        balanco = (alvo - self._mola.atualizar(alvo, dt)) * BALANCO_CABELO

        vista = self._decidir_vista(p)
        if vista:
            cabeca = self.cabecas[vista]
        else:
            cabeca = self.cabecas["cabeca"].copia()

            # Sobrancelhas antes dos olhos: o recorte leva pele de baixo
            # junto, entao subir o sprite cobre a sobrancelha original - e o
            # sprite do olho, colado depois, esconde a emenda de baixo.
            self._colar(cabeca, "sobrancelhas:sobr_e", dy=-p.brow_l * SOBRANCELHA_MAX)
            self._colar(cabeca, "sobrancelhas:sobr_d", dy=-p.brow_r * SOBRANCELHA_MAX)

            # Olhos: com dois quadros de arte, o meio-termo sai de uma mistura.
            # Espremer o sprite aberto arrastava cabelo e sobrancelha junto.
            for lado, ab in (("olho_e", p.eye_open_l), ("olho_d", p.eye_open_r)):
                if ab < 0.12:
                    self._colar(cabeca, f"olhos_fechados:{lado}")
                elif ab > 0.88:
                    self._colar(cabeca, f"olhos_abertos:{lado}")
                else:
                    self._mesclar(cabeca, f"olhos_fechados:{lado}",
                                  f"olhos_abertos:{lado}", ab)
            # Boca: o sprite alvo e MISTURADO com a boca fechada conforme a
            # abertura. Trocando em degrau, uma boca levemente aberta ja
            # escancarava o sprite inteiro - falar baixinho virava bocejo.
            sprite = VISEME_SPRITE.get(p.viseme, "boca_fechada")
            if sprite == "boca_fechada":
                self._colar(cabeca, "boca_fechada:boca")
            else:
                forca = p.smile if sprite.startswith("sorriso") else p.mouth_open
                t = min(1.0, max(0.0, forca * 1.7))
                if t > 0.97:
                    self._colar(cabeca, f"{sprite}:boca")
                else:
                    self._mesclar(cabeca, "boca_fechada:boca", f"{sprite}:boca", t)

            # o cilindro vem DEPOIS de olhos e boca, para a expressao girar
            # junto com o rosto em vez de ficar colada de frente
            yaw = p.head_turn * GIRO_MAX
            pitch = p.head_nod * ACENO_MAX
            if abs(yaw) > 1e-3 or abs(pitch) > 1e-3 or abs(balanco) > 0.05:
                mx, my = self._mapa_cilindro(yaw, pitch, balanco)
                cabeca = cabeca.remapear(mx, my)

        bob = (p.bounce - 0.5) * 4

        # --- tronco ---------------------------------------------------------
        # O corpo acompanha a cabeca em escala reduzida: inclina a partir da
        # cintura, desloca de leve e respira. E a cabeca herda ESTA
        # transformacao antes de aplicar a dela, entao as duas camadas andam
        # juntas e o furo do corpo nunca fica a mostra.
        corpo_afim = self._afim(
            pivo=(self.pivo_corpo[0] + self.bx0, self.pivo_corpo[1] + self.by0),
            graus=-p.head_tilt * 3.5 - p.head_turn * 1.5,
            dx=p.head_x * 11 + p.head_turn * 4,
            dy=p.head_y * 7 + bob * 0.7,
            escala_y=1.0 + (p.bounce - 0.5) * 0.012,    # respiracao
        )

        # --- cabeca, relativa ao tronco -------------------------------------
        squash = 1.0 if vista else 1.0 - 0.10 * abs(p.head_nod)
        cabeca_afim = corpo_afim @ self._afim(
            pivo=(self.pivo[0] + self.hx0, self.pivo[1] + self.hy0),
            graus=-p.head_tilt * 9,
            dx=p.head_x * 20 + p.head_turn * 8,
            # O aceno NAO translada a cabeca: descer a cabeca inteira a
            # descolava do pescoco. Ele sai do encurtamento em torno do pivo.
            dy=p.head_y * 16 + bob * 0.5,
            escala_y=squash,
        )

        quadro = self.plano.copy()
        if self.corpo is not None:
            alt, larg = self.corpo.forma
            self.corpo.transformar(
                self._local(corpo_afim, self.bx0, self.by0), (larg, alt)
            ).sobre_fundo(quadro, self.bx0, self.by0)

        alt, larg = cabeca.forma
        cabeca.transformar(
            self._local(cabeca_afim, self.hx0, self.hy0), (larg, alt)
        ).sobre_fundo(quadro, self.hx0, self.hy0)
        return quadro

    # ------------------------------------------------- transformacoes afins
    @staticmethod
    def _afim(pivo, graus, dx, dy, escala_y=1.0):
        """Matriz 3x3 em coordenadas do quadro (rotacao no pivo + escala + translacao)."""
        m = np.eye(3, dtype=np.float32)
        m[:2] = cv2.getRotationMatrix2D(pivo, graus, 1.0)
        if abs(escala_y - 1.0) > 1e-4:
            m[1, :] *= escala_y
            m[1, 2] += pivo[1] * (1 - escala_y)
        m[0, 2] += dx
        m[1, 2] += dy
        return m

    @staticmethod
    def _local(afim, ox, oy):
        """Converte uma afim do quadro para as coordenadas de uma camada
        recortada em (ox, oy), e inverte: warpAffine quer o mapa destino->origem."""
        desloca = np.array([[1, 0, ox], [0, 1, oy], [0, 0, 1]], np.float32)
        volta = np.array([[1, 0, -ox], [0, 1, -oy], [0, 0, 1]], np.float32)
        return (volta @ afim @ desloca)[:2]
