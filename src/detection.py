"""
Deteção de estrelas: limiar k*ruído, componentes ligados, centróide e filtro de forma.
Para fotos reais usar find_stars(gray).
"""
import cv2
import numpy as np
import pandas as pd
from preprocessing import noise_level


def detect_stars(img: np.ndarray, k: float = 5.0, min_area: int = 3,
                 max_area: float | None = None, max_elong: float = 2.5,
                 noise: float | None = None) -> pd.DataFrame:
    """Estrelas da mais brilhante para a menos.
    max_elong: eixo maior / eixo menor máximo (ramos e aviões dão muito mais)."""
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
        # momentos de 2.ª ordem -> elipse
        sxx = np.bincount(lab, weights=w * xs * xs, minlength=n) / flux - x ** 2
        syy = np.bincount(lab, weights=w * ys * ys, minlength=n) / flux - y ** 2
        sxy = np.bincount(lab, weights=w * xs * ys, minlength=n) / flux - x * y
        tr, dif = sxx + syy, np.sqrt((sxx - syy) ** 2 + 4 * sxy ** 2)
        elong = np.sqrt((tr + dif) / np.maximum(tr - dif, 1e-6))

    out = pd.DataFrame({"x": x, "y": y, "flux": flux, "peak": peak,
                        "area": area, "elong": elong})[1:]          # sem o fundo
    keep = (out["area"] >= min_area) & (out["elong"] <= max_elong)
    if max_area is not None:
        keep &= out["area"] <= max_area
    return out[keep].sort_values("flux", ascending=False).reset_index(drop=True)


def find_stars(gray: np.ndarray, r_frac: float = 0.004, use_sky_mask: bool = True,
               scales=(1, 2, 4), rank: str = "flux") -> pd.DataFrame:
    """top-hat em 3 escalas -> deteção -> tirar repetidas -> só o que está no céu."""
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
        for i in range(len(det)):              # supressão de não-máximos
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
