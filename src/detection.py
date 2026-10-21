"""
Deteção de estrelas numa imagem já pré-processada (fundo ~0).

Limiar: fica só o que está k vezes acima do ruído.
Componentes ligados: cada mancha de píxeis vizinhos é uma estrela candidata.
Centróide ponderado pelo brilho: posição da estrela com precisão abaixo do píxel.

Filtro de forma (necessário nas fotos reais): uma estrela é uma mancha PEQUENA e
REDONDA. Árvores, montanhas, luzes no horizonte e nuvens iluminadas dão manchas
grandes ou alongadas, que no ranking por brilho total passavam à frente das estrelas.

Para fotos reais usar find_stars(gray), que junta tudo: top-hat (preprocessing),
máscara do céu (segmentation) e esta deteção.
"""
import cv2
import numpy as np
import pandas as pd
from preprocessing import noise_level


def detect_stars(img: np.ndarray, k: float = 5.0, min_area: int = 3,
                 max_area: float | None = None, max_elong: float = 2.5,
                 noise: float | None = None) -> pd.DataFrame:
    """Devolve as estrelas detetadas, da mais brilhante para a menos (brilho total).

    max_area:      área máxima de uma estrela em píxeis (None = sem limite).
    max_elong:     razão máxima entre o eixo maior e o menor da mancha
                   (1 = círculo perfeito; um traço de avião ou um ramo dá muito mais).
    noise:         nível de ruído; se None, é medido em img. Dá-lo explicitamente quando
                   img já não tem fundo (ex.: depois de um top-hat, a imagem é quase toda
                   zero e o MAD daria ~0, pondo o limiar em zero).
    """
    if noise is None:
        noise = noise_level(img)
    mask = img > k * noise
    n, labels = cv2.connectedComponents(mask.astype(np.uint8), connectivity=8)

    lab = labels.ravel()
    w = np.where(mask, img, 0).ravel().astype(np.float64)
    ys, xs = np.indices(img.shape)
    xs, ys = xs.ravel().astype(np.float64), ys.ravel().astype(np.float64)

    flux = np.bincount(lab, weights=w, minlength=n)
    area = np.bincount(lab, minlength=n)
    peak = np.zeros(n)
    np.maximum.at(peak, lab, w)
    with np.errstate(invalid="ignore", divide="ignore"):
        x = np.bincount(lab, weights=w * xs, minlength=n) / flux
        y = np.bincount(lab, weights=w * ys, minlength=n) / flux
        # momentos de 2.ª ordem -> forma da mancha (como uma elipse)
        sxx = np.bincount(lab, weights=w * xs * xs, minlength=n) / flux - x ** 2
        syy = np.bincount(lab, weights=w * ys * ys, minlength=n) / flux - y ** 2
        sxy = np.bincount(lab, weights=w * xs * ys, minlength=n) / flux - x * y
        # valores próprios da matriz de covariância = variância ao longo dos eixos da elipse
        tr, dif = sxx + syy, np.sqrt((sxx - syy) ** 2 + 4 * sxy ** 2)
        elong = np.sqrt((tr + dif) / np.maximum(tr - dif, 1e-6))

    out = pd.DataFrame({"x": x, "y": y, "flux": flux, "peak": peak,
                        "area": area, "elong": elong})[1:]          # [1:] tira o fundo (label 0)
    keep = (out["area"] >= min_area) & (out["elong"] <= max_elong)
    if max_area is not None:
        keep &= out["area"] <= max_area
    return out[keep].sort_values("flux", ascending=False).reset_index(drop=True)


def find_stars(gray: np.ndarray, r_frac: float = 0.004, use_sky_mask: bool = True,
               scales=(1, 2, 4), rank: str = "flux") -> pd.DataFrame:
    """Pipeline completo da fase 2 para uma imagem em tons de cinzento:
    top-hat (várias escalas) -> deteção -> filtro de forma -> só o que está no céu.

    Multi-escala: em astrofotos processadas as estrelas mais brilhantes têm halos
    grandes (20-40 px), maiores que o disco do top-hat, e eram apagadas como se
    fossem paisagem. Corre-se o top-hat com raios r, 2r, 4r: cada estrela é
    apanhada na escala do seu tamanho. Na mesma estrela vista em várias escalas
    fica só a deteção com mais brilho (supressão de não-máximos).

    A área máxima em cada escala é 2x a do disco: maior que isso não é uma estrela.
    """
    from scipy.spatial import cKDTree
    from preprocessing import tophat
    from segmentation import sky_mask

    parts = []
    for sc in scales:
        img, r = tophat(gray, r_frac * sc)
        d = detect_stars(img, max_area=2 * np.pi * r * r)
        d["r"] = r
        parts.append(d)
    det = pd.concat(parts, ignore_index=True).sort_values("flux", ascending=False)
    det = det.reset_index(drop=True)

    if len(scales) > 1 and len(det):
        xy = det[["x", "y"]].values
        tree = cKDTree(xy)
        keep = np.ones(len(det), bool)
        for i in range(len(det)):              # da mais brilhante para a menos
            if keep[i]:
                for j in tree.query_ball_point(xy[i], det["r"].iat[i]):
                    if j > i:
                        keep[j] = False
        det = det[keep].reset_index(drop=True)

    if use_sky_mask and len(det):
        mask = sky_mask(gray)
        xi = det["x"].round().astype(int).clip(0, gray.shape[1] - 1)
        yi = det["y"].round().astype(int).clip(0, gray.shape[0] - 1)
        det = det[mask[yi, xi]].reset_index(drop=True)

    if rank == "peak":
        det = det.sort_values("peak", ascending=False)
    elif rank == "clip":
        cap = 0.8 * det["peak"].max() if len(det) else 0
        det = det.assign(_k=det["peak"].clip(upper=cap)).sort_values(["_k", "flux"], ascending=False).drop(columns="_k")
    return det.reset_index(drop=True)
