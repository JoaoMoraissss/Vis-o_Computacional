"""
Pose da câmara: para onde aponta (RA, Dec do centro), rotação (roll) e campo de visão (FOV).

Usa exatamente o mesmo modelo de câmara do gerador sintético (project_to_pixels),
por isso uma pose certa põe cada estrela do catálogo em cima da estrela da foto,
mesmo nos cantos de uma imagem com 60° de campo, onde a projeção deforma muito.

Uma foto tem uma só câmara: se acertarmos a pose, ficam identificadas de uma vez
TODAS as estrelas e constelações da imagem.
"""
import numpy as np
import pandas as pd
from scipy.optimize import least_squares
from scipy.spatial import cKDTree

from synthetic import project_to_pixels


def inverse_gnomonic(xi, eta, ra0_deg, dec0_deg):
    """Inverso da gnomónica: de (xi, eta) no plano tangente para (RA, Dec) em graus."""
    ra0, dec0 = np.radians(ra0_deg), np.radians(dec0_deg)
    rho = np.hypot(xi, eta)
    c = np.arctan(rho)
    with np.errstate(invalid="ignore", divide="ignore"):
        dec = np.arcsin(np.cos(c) * np.sin(dec0) + np.where(rho > 0, eta * np.sin(c) * np.cos(dec0) / rho, 0))
        ra = ra0 + np.arctan2(xi * np.sin(c), rho * np.cos(dec0) * np.cos(c) - eta * np.sin(dec0) * np.sin(c))
    return np.degrees(ra) % 360, np.degrees(dec)


def initial_pose(T_inv, angle_deg, scale, plane_center, img_shape):
    """Primeira estimativa da pose a partir da semelhança do passo 4.

    T_inv:        leva píxeis para o plano tangente da constelação
    angle_deg:    rotação da semelhança (plano -> píxeis)
    scale:        píxeis por unidade do plano ≈ distância focal f em píxeis
    plane_center: (RA, Dec) do centro do plano tangente da constelação
    """
    h, w = img_shape
    xi, eta = T_inv(np.array([[w / 2, h / 2]]))[0]
    ra0, dec0 = inverse_gnomonic(xi, eta, *plane_center)
    fov = np.degrees(2 * np.arctan((w / 2) / scale))
    # project_to_pixels faz píxel = centro + f * R(roll + 180°) * (xi, eta)
    roll = (angle_deg - 180) % 360
    return np.array([ra0, dec0, roll, fov])


def project(stars: pd.DataFrame, pose, img_shape):
    ra0, dec0, roll, fov = pose
    h, w = img_shape
    x, y, inside = project_to_pixels(stars, ra0, dec0, fov, roll, w, h)
    return np.column_stack([x, y]), inside


def refine_pose(pose, stars: pd.DataFrame, pix: np.ndarray, img_shape):
    """Ajusta a pose por mínimos quadrados para as estrelas `stars` caírem nos píxeis `pix`."""
    def residuals(p):
        xy, _ = project(stars, p, img_shape)
        return (xy - pix).ravel()

    p0 = np.array(pose, float)
    p0[2] = p0[2] % 360
    res = least_squares(residuals, p0, x_scale=[1, 1, 1, 1], loss="soft_l1", f_scale=3.0)
    return res.x


def match_catalog(pose, catalog: pd.DataFrame, det_xy: np.ndarray, img_shape, tol_px: float,
                  n_brightest: int | None = None):
    """Projeta o catálogo com a pose e associa cada deteção à estrela projetada mais próxima.

    Só aceita pares mútuos (a deteção mais próxima da estrela e vice-versa) e a menos de tol_px.
    n_brightest: se dado, usa só as n estrelas do catálogo mais brilhantes dentro da imagem.
      Serve para pontuar poses de forma justa: sem isto, uma pose com FOV enorme mete
      milhares de estrelas na imagem e acerta em muitas deteções só por acaso.
    Devolve (índices das deteções, índices no catálogo).
    """
    xy, inside = project(catalog, pose, img_shape)
    cat_idx = np.where(inside)[0]
    if n_brightest is not None:
        cat_idx = cat_idx[np.argsort(catalog["mag"].values[cat_idx])[:n_brightest]]
    if len(cat_idx) == 0 or len(det_xy) == 0:
        return np.array([], int), np.array([], int)
    cat_xy = xy[cat_idx]
    d1, j = cKDTree(cat_xy).query(det_xy)                # deteção -> estrela
    _, back = cKDTree(det_xy).query(cat_xy[j])           # essa estrela -> deteção
    ok = (d1 < tol_px) & (back == np.arange(len(det_xy)))
    return np.where(ok)[0], cat_idx[j[ok]]
