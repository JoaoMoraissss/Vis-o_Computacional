"""
Desenhar o resultado por cima da foto: linhas e nomes das constelações,
estrelas identificadas e a pose da câmara.

Como a pose dá a posição de QUALQUER estrela do catálogo na imagem, as linhas
são desenhadas a partir do catálogo e não das deteções. Se uma estrela da linha
não foi detetada (fraca, atrás de uma árvore), a linha aparece na mesma.
Isto é também a melhor verificação visual: se a pose estiver errada, as linhas
não caem em cima das estrelas.
"""
import cv2
import numpy as np
import pandas as pd

from synthetic import gnomonic, project_to_pixels

LINE_COLOR = (255, 190, 90)       # BGR: azul claro
NAME_COLOR = (90, 220, 255)       # amarelo
STAR_COLOR = (80, 255, 80)        # verde
OTHER_COLOR = (150, 120, 90)      # constelações visíveis mas não reportadas


def draw_result(img: np.ndarray, result: dict, stars: pd.DataFrame, consts: dict,
                show_all: bool = True, star_names_mag: float = 2.0,
                all_stars: bool = False, sky: np.ndarray | None = None) -> np.ndarray:
    """Devolve uma cópia de img (BGR ou cinzento) com o resultado desenhado.

    result:         saída de matching.identify (precisa de "pose", "stars", "constellations")
    stars, consts:  catálogo (load_stars, load_constellations)
    show_all:       desenha também, mais ténues, as outras constelações que caem na imagem
    star_names_mag: escreve o nome próprio das estrelas identificadas até esta magnitude
    all_stars:      marca TODAS as estrelas identificadas (útil para depurar);
                    por defeito só as que fazem parte das linhas das constelações
    sky:            máscara do céu (segmentation.sky_mask); se dada, as linhas não são
                    desenhadas por cima da paisagem
    """
    out = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR) if img.ndim == 2 else img.copy()
    h, w = out.shape[:2]
    u = w / 1000                                  # unidade de espessura/tamanho, cresce com a imagem
    thick = max(1, round(1.5 * u))
    font = cv2.FONT_HERSHEY_SIMPLEX
    ra0, dec0, roll, fov = result["pose"]

    x, y, inside = project_to_pixels(stars, ra0, dec0, fov, roll, w, h)
    front = gnomonic(stars["ra_deg"].values, stars["dec_deg"].values, ra0, dec0)[2]
    hips = stars["hip"].fillna(-1).astype(int).values
    pos = {hp: (x[i], y[i], inside[i], front[i]) for i, hp in enumerate(hips)}

    def draw_constellation(abbr, color, width, label):
        drawn = []
        for line in consts[abbr]["lines"]:
            for a, b in zip(line[:-1], line[1:]):
                if a not in pos or b not in pos:
                    continue
                pa, pb = pos[a], pos[b]
                if not (pa[3] and pb[3]) or not (pa[2] or pb[2]):   # atrás da câmara / fora
                    continue
                cv2.line(out, (int(pa[0]), int(pa[1])), (int(pb[0]), int(pb[1])),
                         color, width, cv2.LINE_AA)
                drawn += [p[:2] for p in (pa, pb) if p[2]]
        if label and drawn:
            cx, cy = np.mean(drawn, axis=0)
            _put_label(out, consts[abbr]["name"], (int(cx), int(cy)), 0.9 * u, color, font)

    found = result["constellations"]
    if show_all:
        for abbr in consts:
            if abbr not in found:
                draw_constellation(abbr, OTHER_COLOR, max(1, thick // 2), label=False)
    for abbr in found:
        draw_constellation(abbr, LINE_COLOR, thick, label=False)
    if sky is not None:                           # repõe a foto original onde não é céu
        base = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR) if img.ndim == 2 else img
        out[~sky] = base[~sky]

    # estrelas identificadas: círculo verde; nome das mais brilhantes
    mag = pd.Series(stars["mag"].values, index=hips)
    mag = mag[~mag.index.duplicated()]
    in_lines = {h for line_set in consts.values() for line in line_set["lines"] for h in line}
    for _, s in result["stars"].iterrows():
        if not all_stars and s["hip"] not in in_lines:
            continue
        c = (int(s["x"]), int(s["y"]))
        cv2.circle(out, c, max(4, round(7 * u)), STAR_COLOR, max(1, thick // 2), cv2.LINE_AA)
        if isinstance(s["proper"], str) and mag.get(s["hip"], 99) <= star_names_mag:
            _put_label(out, s["proper"], (c[0] + round(10 * u), c[1] - round(6 * u)),
                       0.6 * u, STAR_COLOR, font, center=False)

    # nomes das constelações por cima de tudo
    for abbr in found:
        pts = [pos[h][:2] for h in found[abbr] if h in pos]
        if pts:
            cx, cy = np.mean(pts, axis=0)
            _put_label(out, consts[abbr]["name"], (int(cx), int(cy)), 1.1 * u, NAME_COLOR, font)

    info = (f"centro RA {ra0:.1f}  Dec {dec0:+.1f}   rot {roll:.0f}   FOV {fov:.0f} graus   "
            f"score {result['score']:.2f}   {len(result['stars'])} estrelas")
    _put_label(out, info, (round(12 * u), round(30 * u)), 0.7 * u, (255, 255, 255), font, center=False)
    return out


def draw_failure(img: np.ndarray, msg: str = "sem solucao") -> np.ndarray:
    out = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR) if img.ndim == 2 else img.copy()
    u = out.shape[1] / 1000
    _put_label(out, msg, (round(12 * u), round(30 * u)), 0.9 * u, (80, 80, 255),
               cv2.FONT_HERSHEY_SIMPLEX, center=False)
    return out


def _put_label(img, text, org, scale, color, font, center=True):
    """Texto com contorno preto, para se ler sobre céu claro ou escuro.
    (O OpenCV não desenha acentos, por isso os nomes ficam sem eles.)"""
    th = max(1, round(scale * 2))
    (tw, tht), _ = cv2.getTextSize(text, font, scale, th)
    x, y = org
    if center:
        x, y = x - tw // 2, y + tht // 2
    cv2.putText(img, text, (x, y), font, scale, (0, 0, 0), th + 3, cv2.LINE_AA)
    cv2.putText(img, text, (x, y), font, scale, color, th, cv2.LINE_AA)
