"""
Desenhar o resultado: linhas e nomes das constelações, estrelas e pose.
As linhas vêm do catálogo projetado, por isso aparecem mesmo com estrelas tapadas.
"""
import cv2
import numpy as np
import pandas as pd

from synthetic import gnomonic
from projection import project

LINE_COLOR = (255, 190, 90)       # BGR: azul claro
NAME_COLOR = (90, 220, 255)       # amarelo
STAR_COLOR = (80, 255, 80)        # verde
OTHER_COLOR = (150, 120, 90)      # visíveis mas não reportadas


def draw_result(img: np.ndarray, result: dict, stars: pd.DataFrame, consts: dict,
                show_all: bool = True, star_names_mag: float = 2.0,
                all_stars: bool = False, sky: np.ndarray | None = None) -> np.ndarray:
    """Cópia de img com o resultado. sky = máscara para não desenhar sobre a paisagem."""
    out = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR) if img.ndim == 2 else img.copy()
    h, w = out.shape[:2]
    u = w / 1000                                  # escala com o tamanho da imagem
    thick = max(1, round(1.5 * u))
    font = cv2.FONT_HERSHEY_SIMPLEX
    ra0, dec0, roll, fov = result["pose"][:4]

    xy, inside = project(stars, result["pose"], (h, w))
    x, y = xy[:, 0], xy[:, 1]
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
                if not (pa[3] and pb[3]) or not (pa[2] or pb[2]):
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
    if sky is not None:
        base = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR) if img.ndim == 2 else img
        out[~sky] = base[~sky]

    # estrelas identificadas
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

    # nomes por cima de tudo
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
    """Texto com contorno preto (o OpenCV não faz acentos)."""
    th = max(1, round(scale * 2))
    (tw, tht), _ = cv2.getTextSize(text, font, scale, th)
    x, y = org
    if center:
        x, y = x - tw // 2, y + tht // 2
    cv2.putText(img, text, (x, y), font, scale, (0, 0, 0), th + 3, cv2.LINE_AA)
    cv2.putText(img, text, (x, y), font, scale, color, th, cv2.LINE_AA)
