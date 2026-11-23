"""
Matching por triângulos medidos na esfera (a versão usada).

No plano os triângulos deformavam-se nas grandes angulares; aqui os lados são
ângulos no céu. Para cada FOV suposto os píxeis viram raios 3-D, procuram-se
triângulos com a mesma forma e tamanho e a rotação sai por Kabsch.
"""
from itertools import combinations

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

import matching as M
import projection as P
from catalog import CATALOG_DIR, load_stars, load_constellations

SPHERE_NPZ = CATALOG_DIR / "triangles_sphere.npz"
FOV_GUESSES = np.geomspace(8, 100, 20)          # ~14% entre cada (até 150° estragava uma foto de desenvolvimento)


def radec_to_vec(ra_deg, dec_deg) -> np.ndarray:
    """(RA, Dec) em graus -> vetores unitários (n, 3)."""
    ra, dec = np.radians(np.asarray(ra_deg, float)), np.radians(np.asarray(dec_deg, float))
    return np.column_stack([np.cos(dec) * np.cos(ra), np.cos(dec) * np.sin(ra), np.sin(dec)])


def vec_to_radec(v):
    v = v / np.linalg.norm(v)
    return np.degrees(np.arctan2(v[1], v[0])) % 360, np.degrees(np.arcsin(v[2]))


def triangles_from_angles(A: np.ndarray, combos: np.ndarray):
    """Descritor, lado maior e vértices ordenados de vários triângulos de uma vez.
    A = matriz de ângulos entre estrelas."""
    i, j, k = combos.T
    sides = np.column_stack([A[j, k], A[i, k], A[i, j]])
    order = np.argsort(-sides, axis=1)
    s = np.take_along_axis(sides, order, axis=1)
    desc = s[:, 1:] / s[:, :1]
    ok = desc[:, 1] >= M.MIN_RATIO
    return desc[ok], s[ok, 0], np.take_along_axis(combos, order, axis=1)[ok]


def angle_matrix(V: np.ndarray) -> np.ndarray:
    return np.arccos(np.clip(V @ V.T, -1, 1))


def build_sphere_index(max_stars: int = 10):
    """Triângulos (em ângulos) das estrelas mais brilhantes de cada constelação."""
    stars, consts = load_stars(), load_constellations()
    descs, sizes, hips, cons = [], [], [], []
    for abbr in consts:
        sel = M.constellation_stars(abbr, stars, consts, max_stars)
        if len(sel) < 3:
            continue
        V = radec_to_vec(sel["ra_deg"], sel["dec_deg"])
        combos = np.array(list(combinations(range(len(sel)), 3)))
        d, a, ids = triangles_from_angles(angle_matrix(V), combos)
        descs.append(d); sizes.append(a)
        hips.append(sel["hip"].values.astype(int)[ids]); cons += [abbr] * len(d)
    np.savez(SPHERE_NPZ, desc=np.vstack(descs), a=np.concatenate(sizes),
             hips=np.vstack(hips), con=np.array(cons))


def load_sphere_index(bright_mag: float = 5.0) -> dict:
    """Índice antigo + triângulos esféricos + vetores de todas as estrelas."""
    index = M.load_catalog_index()
    if not SPHERE_NPZ.exists():
        build_sphere_index()
    data = np.load(SPHERE_NPZ)
    index.update({f"s_{k}": data[k] for k in data.files})
    index["s_tree"] = cKDTree(index["s_desc"])

    stars = index.setdefault("stars", load_stars())
    index["vec"] = radec_to_vec(stars["ra_deg"], stars["dec_deg"])
    hips = stars["hip"].fillna(-1).astype(int).values
    row = pd.Series(np.arange(len(stars)), index=hips)
    index["s_rows"] = row[~row.index.duplicated()].reindex(index["s_hips"].ravel()).values.reshape(-1, 3)
    mag = stars["mag"].values
    index["s_maxmag"] = mag[index["s_rows"]].max(1)
    by_mag = np.argsort(mag)
    index["bright_rows"] = by_mag[mag[by_mag] <= bright_mag]
    return index


def focal(F_deg: float, w: int, lens: str = "rect") -> float:
    """Focal em píxeis para um FOV horizontal."""
    if lens == "fish":
        return (w / 2) / np.radians(F_deg / 2)          # r = f·θ
    return (w / 2) / np.tan(np.radians(F_deg) / 2)      # r = f·tan(θ)


def fov_from_focal(f: float, w: int, lens: str = "rect") -> float:
    if lens == "fish":
        return np.degrees(2 * (w / 2) / f)
    return np.degrees(2 * np.arctan((w / 2) / f))


def pixel_rays(xy: np.ndarray, f: float, w: int, h: int, lens: str = "rect") -> np.ndarray:
    """Píxeis -> raios unitários da câmara (pinhole ou olho-de-peixe)."""
    u, v = xy[:, 0] - w / 2, xy[:, 1] - h / 2
    if lens == "fish":
        th, phi = np.hypot(u, v) / f, np.arctan2(v, u)
        return np.column_stack([np.sin(th) * np.cos(phi), np.sin(th) * np.sin(phi), np.cos(th)])
    D = np.column_stack([u, v, np.full(len(xy), f)])
    return D / np.linalg.norm(D, axis=1, keepdims=True)


def kabsch(V: np.ndarray, D: np.ndarray) -> np.ndarray:
    """Rotação que leva V (céu) a D (câmara), D ≈ V @ R.T."""
    U, _, Vt = np.linalg.svd(V.T @ D)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    return Vt.T @ np.diag([1, 1, d]) @ U.T


def project_rows(R, f, rows, vec, w, h, lens: str = "rect"):
    """Projeta as estrelas rows com (R, f)."""
    Pc = vec[rows] @ R.T
    z = Pc[:, 2]
    with np.errstate(divide="ignore", invalid="ignore"):
        if lens == "fish":
            th = np.arccos(np.clip(z, -1, 1))
            rxy = np.hypot(Pc[:, 0], Pc[:, 1])
            k = np.where(rxy > 1e-12, f * th / rxy, 0.0)
            x, y = w / 2 + k * Pc[:, 0], h / 2 + k * Pc[:, 1]
        else:
            x = w / 2 + f * Pc[:, 0] / z
            y = h / 2 + f * Pc[:, 1] / z
    inside = (z > 0) & (x >= 0) & (x < w) & (y >= 0) & (y < h)
    return np.column_stack([x, y]), inside


def score_camera(R, f, index, top_xy, img_shape, tol_px, n_brightest, lens: str = "rect"):
    """Fração das deteções de topo que caem em cima de uma estrela brilhante do catálogo."""
    h, w = img_shape
    rows = index["bright_rows"]
    xy, inside = project_rows(R, f, rows, index["vec"], w, h, lens)
    sel = np.where(inside)[0][:n_brightest]
    if len(sel) < 3:
        return 0.0, []
    d, j = cKDTree(xy[sel]).query(top_xy)
    ok = d < tol_px
    pairs = {}
    for di in np.where(ok)[0]:                   # cada estrela só conta uma vez
        pairs.setdefault(int(j[di]), int(di))
    return len(pairs) / len(top_xy), [(di, int(rows[sel[k]])) for k, di in pairs.items()]


def refine_camera(pairs, det_xy, index, f, img_shape, span: float = 0.15, steps: int = 31,
                  lens: str = "rect"):
    """Afina f em ±15% e a rotação, porque o FOV da grelha é só aproximado."""
    h, w = img_shape
    di = np.array([p[0] for p in pairs]); V = index["vec"][[p[1] for p in pairs]]
    best = None
    for fs in f * np.linspace(1 - span, 1 + span, steps):
        D = pixel_rays(det_xy[di], fs, w, h, lens)
        R = kabsch(V, D)
        err = (1 - (V @ R.T * D).sum(1)).sum()
        if best is None or err < best[0]:
            best = (err, R, fs)
    return best[1], best[2]


def camera_to_pose(R, f, pairs, det_xy, index, img_shape, lens: str = "rect"):
    """(R, f) -> pose (ra0, dec0, roll, fov) de projection.py, e refina."""
    h, w = img_shape
    ra0, dec0 = vec_to_radec(R.T @ np.array([0.0, 0.0, 1.0]))
    fov = fov_from_focal(f, w, lens)
    extra = [P.FISHEYE] if lens == "fish" else []
    di = np.array([p[0] for p in pairs]); rows = np.array([p[1] for p in pairs])
    cat = index["stars"].iloc[rows]
    errs = [np.abs(P.project(cat, [ra0, dec0, r, fov] + extra, img_shape)[0] - det_xy[di]).sum()
            for r in range(0, 360, 5)]
    pose = np.array([ra0, dec0, 5.0 * int(np.argmin(errs)), fov] + extra)
    return P.refine_pose(pose, cat, det_xy[di], img_shape)


def identify_sphere(det: pd.DataFrame, index: dict, img_shape, n_stars: int = 20,
                    eps: float = 0.01, size_tol: float = 0.15, tol_frac: float = 0.006,
                    n_score: int = 30, min_score: float = 0.5, stop_score: float = 0.9,
                    max_hyp: int = 5000, min_con_stars: int = 4, fov_guesses=FOV_GUESSES,
                    coarse_mult: float = 3.0, coarse_min: float = 0.15,
                    weak_score: float = 0.3, ratio: float = 1.8, schedule: str = "suave",
                    lens: str = "rect"):
    """Identifica a pose e as constelações. Devolve {} se não houver solução."""
    h, w = img_shape
    det_xy = det[["x", "y"]].values
    n = min(n_stars, len(det))
    if n < 3:
        return {}
    tol_px = tol_frac * max(h, w)
    top_xy = det_xy[:min(n_score, len(det))]
    combos = np.array(list(combinations(range(n), 3)))

    # candidatos para cada FOV suposto (forma e tamanho)
    hyps, rays = [], {}
    for F in fov_guesses:
        f = focal(F, w, lens)
        D = pixel_rays(det_xy[:n], f, w, h, lens)
        rays[F] = (f, D)
        desc, a, ids = triangles_from_angles(angle_matrix(D), combos)
        for i, js in enumerate(index["s_tree"].query_ball_point(desc, eps)):
            if not js:
                continue
            js = np.array(js)
            js = js[np.abs(a[i] / index["s_a"][js] - 1) < size_tol]
            for j in js:
                hyps.append((ids[i].max(), index["s_maxmag"][j], F, tuple(ids[i]), j))

    # estrelas mais brilhantes primeiro; contagem larga -> afinar -> contagem final
    hyps.sort(key=lambda t: (t[0], t[1]))
    best = None
    runner_up = 0.0           # melhor pose diferente (eixo a mais de 5°)
    nb = 3 * len(top_xy)
    for _, _, F, tri, j in hyps[:max_hyp]:
        f, D = rays[F]
        R = kabsch(index["vec"][index["s_rows"][j]], D[list(tri)])
        coarse, pairs = score_camera(R, f, index, top_xy, img_shape, coarse_mult * tol_px, nb, lens)
        if coarse < coarse_min:
            continue
        for _ in range(2):
            R, f = refine_camera(pairs, det_xy, index, f, img_shape, lens=lens)
            score, pairs = score_camera(R, f, index, top_xy, img_shape, tol_px, nb, lens)
            if len(pairs) < 3:
                break
        axis = R.T @ np.array([0.0, 0.0, 1.0])
        if best is None or score > best[0]:
            if best is not None and best[4] @ axis < np.cos(np.radians(5)):
                runner_up = max(runner_up, best[0])
            best = (score, R, f, pairs, axis)
        elif best[4] @ axis < np.cos(np.radians(5)):
            runner_up = max(runner_up, score)
        if score >= stop_score:
            break

    # aceita se o score for alto ou se ganhar bem à 2.ª melhor (teste da razão)
    if best is None:
        return {}
    if best[0] < min_score and not (best[0] >= weak_score and best[0] >= ratio * max(runner_up, 1e-9)):
        return {}
    score, R, f, pairs, _ = best
    pose = camera_to_pose(R, f, pairs, det_xy, index, img_shape, lens)
    if not (1.0 < pose[3] < 120):
        return {}
    return M.finalize(pose, score, det_xy, index, img_shape, tol_px, min_con_stars, schedule=schedule)


if __name__ == "__main__":
    build_sphere_index()
    idx = load_sphere_index()
    print(f"{len(idx['s_desc'])} triângulos esféricos guardados em {SPHERE_NPZ.name}")
    print()
    M.evaluate_identification(identify_fn=identify_sphere, index=idx)
