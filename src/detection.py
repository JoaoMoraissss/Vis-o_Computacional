"""
Deteção de estrelas numa imagem já pré-processada (fundo ~0).

Limiar: fica só o que está k vezes acima do ruído.
Componentes ligados: cada mancha de píxeis vizinhos é uma estrela candidata.
Centróide ponderado pelo brilho: posição da estrela com precisão abaixo do píxel.
"""
import cv2
import numpy as np
import pandas as pd
from preprocessing import noise_level

def detect_stars(img: np.ndarray, k: float = 5.0, min_area: int = 3) -> pd.DataFrame:
    mask = img > k * noise_level(img)
    n, labels = cv2.connectedComponents(mask.astype(np.uint8), connectivity=8)

    lab = labels.ravel()
    w = np.where(mask, img, 0).ravel().astype(np.float64)       
    ys, xs = np.indices(img.shape)
    flux = np.bincount(lab, weights=w, minlength=n)
    area = np.bincount(lab, minlength=n)
    with np.errstate(invalid="ignore", divide="ignore"):
        x = np.bincount(lab, weights=w * xs.ravel(), minlength=n) / flux
        y = np.bincount(lab, weights=w * ys.ravel(), minlength=n) / flux

    out = pd.DataFrame({"x": x, "y": y, "flux": flux, "area": area})[1:]   
    out = out[out["area"] >= min_area]
    return out.sort_values("flux", ascending=False).reset_index(drop=True)