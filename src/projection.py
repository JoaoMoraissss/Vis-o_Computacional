"""
Pose da câmara (RA, Dec do centro, roll, FOV): projetar o catálogo e ajustar.
Usa o mesmo modelo de câmara do gerador sintético.
"""
import numpy as np
import pandas as pd
from scipy.optimize import least_squares
from scipy.spatial import cKDTree

from synthetic import gnomonic, project_to_pixels


def inverse_gnomonic(xi, eta, ra0_deg, dec0_deg):
    """(xi, eta) -> (RA, Dec)."""
    ra0, dec0 = np.radians(ra0_deg), np.radians(dec0_deg)
    rho = np.hypot(xi, eta)
    c = np.arctan(rho)
    with np.errstate(invalid="ignore", divide="ignore"):
        dec = np.arcsin(np.cos(c) * np.sin(dec0) + np.where(rho > 0, eta * np.sin(c) * np.cos(dec0) / rho, 0))
        ra = ra0 + np.arctan2(xi * np.sin(c), rho * np.cos(dec0) * np.cos(c) - eta * np.sin(dec0) * np.sin(c))
    return np.degrees(ra) % 360, np.degrees(dec)


def initial_pose(T_inv, angle_deg, scale, plane_center, img_shape):
    """Pose inicial a partir da semelhança no plano (versão antiga). scale ≈ focal."""
    h, w = img_shape
    xi, eta = T_inv(np.array([[w / 2, h / 2]]))[0]
    ra0, dec0 = inverse_gnomonic(xi, eta, *plane_center)
    fov = np.degrees(2 * np.arctan((w / 2) / scale))
    roll = (angle_deg - 180) % 360
    return np.array([ra0, dec0, roll, fov])


FISHEYE = 1.0      # marca na pose (5.º valor)


def project(stars: pd.DataFrame, pose, img_shape):
    """Píxeis de cada estrela.
    pose de 4 valores: pinhole; 5: olho-de-peixe; 7: + centro ótico (dx, dy) e distorção k1."""
    ra0, dec0, roll, fov = pose[:4]
    h, w = img_shape
    if len(pose) == 5:
        return _project_fisheye(stars, ra0, dec0, roll, fov, w, h)
    x, y, inside = project_to_pixels(stars, ra0, dec0, fov, roll, w, h)
    if len(pose) == 7:
        dx, dy, k1 = pose[4:]
        u, v = x - w / 2, y - h / 2
        k = 1 + k1 * (u * u + v * v) / ((w / 2) ** 2 + (h / 2) ** 2)
        x, y = w / 2 + dx + u * k, h / 2 + dy + v * k
        front = gnomonic(stars["ra_deg"].values, stars["dec_deg"].values, ra0, dec0)[2]
        inside = front & (x >= 0) & (x < w) & (y >= 0) & (y < h)
    return np.column_stack([x, y]), inside


def _project_fisheye(stars, ra0, dec0, roll, fov, w, h):
    """Como project_to_pixels mas com r = f·θ."""
    xi, eta, front = gnomonic(stars["ra_deg"].values, stars["dec_deg"].values, ra0, dec0)
    rho = np.hypot(xi, eta)                         # tan(θ)
    with np.errstate(divide="ignore", invalid="ignore"):
        k = np.where(rho > 1e-12, np.arctan(rho) / rho, 1.0)
    f = (w / 2) / np.radians(fov / 2)
    u, v = -f * xi * k, -f * eta * k
    r = np.radians(roll)
    x = w / 2 + np.cos(r) * u - np.sin(r) * v
    y = h / 2 + np.sin(r) * u + np.cos(r) * v
    inside = front & (x >= 0) & (x < w) & (y >= 0) & (y < h)
    return np.column_stack([x, y]), inside


def refine_pose(pose, stars: pd.DataFrame, pix: np.ndarray, img_shape, extended=False):
    """Mínimos quadrados (soft-L1) para as estrelas caírem em pix.
    extended=True também ajusta dx, dy, k1 (precisa de muitas estrelas); "k1" só k1."""
    h, w = img_shape

    def residuals(p):
        xy, _ = project(stars, p, img_shape)
        return (xy - pix).ravel()

    p0 = np.array(pose, float)
    p0[2] = p0[2] % 360
    if len(p0) == 5:
        def res5(q):
            return residuals(np.concatenate([q, [FISHEYE]]))
        q = least_squares(res5, p0[:4], x_scale=[1, 1, 1, 1], loss="soft_l1", f_scale=3.0).x
        return np.concatenate([q, [FISHEYE]])
    if extended == "k1":
        base = p0[:4]
        k0 = p0[6] if len(p0) == 7 else 0.0

        def res_k1(q):
            return residuals(np.concatenate([q[:4], [0.0, 0.0, q[4]]]))
        lo = [-np.inf, -90, -np.inf, 0.1, -0.3]
        hi = [np.inf, 90, np.inf, 179, 0.3]
        q0 = np.clip(np.concatenate([base, [k0]]), np.array(lo) + 1e-9, np.array(hi) - 1e-9)
        q = least_squares(res_k1, q0, x_scale=[1, 1, 1, 1, 0.01], loss="soft_l1",
                          f_scale=3.0, bounds=(lo, hi)).x
        return np.concatenate([q[:4], [0.0, 0.0, q[4]]])
    if extended and len(p0) == 4:
        p0 = np.concatenate([p0, [0.0, 0.0, 0.0]])
    if len(p0) == 7:
        lo = [-np.inf, -90, -np.inf, 0.1, -0.1 * w, -0.1 * h, -0.3]
        hi = [np.inf, 90, np.inf, 179, 0.1 * w, 0.1 * h, 0.3]
        p0 = np.clip(p0, np.array(lo) + 1e-9, np.array(hi) - 1e-9)
        res = least_squares(residuals, p0, x_scale=[1, 1, 1, 1, 10, 10, 0.01],
                            loss="soft_l1", f_scale=3.0, bounds=(lo, hi))
    else:
        res = least_squares(residuals, p0, x_scale=[1, 1, 1, 1], loss="soft_l1", f_scale=3.0)
    return res.x


def match_catalog(pose, catalog: pd.DataFrame, det_xy: np.ndarray, img_shape, tol_px: float,
                  n_brightest: int | None = None):
    """Pares mútuos deteção <-> estrela projetada a menos de tol_px.
    n_brightest limita às estrelas mais brilhantes (senão FOVs enormes acertavam por acaso)."""
    xy, inside = project(catalog, pose, img_shape)
    cat_idx = np.where(inside)[0]
    if n_brightest is not None:
        cat_idx = cat_idx[np.argsort(catalog["mag"].values[cat_idx])[:n_brightest]]
    if len(cat_idx) == 0 or len(det_xy) == 0:
        return np.array([], int), np.array([], int)
    cat_xy = xy[cat_idx]
    d1, j = cKDTree(cat_xy).query(det_xy)
    _, back = cKDTree(det_xy).query(cat_xy[j])
    ok = (d1 < tol_px) & (back == np.arange(len(det_xy)))
    return np.where(ok)[0], cat_idx[j[ok]]
