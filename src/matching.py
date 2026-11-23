"""
Matching por triângulos, 1.ª versão (lados medidos em píxeis).
A versão usada agora é matching_sphere.py, mas o índice, finalize() e as
avaliações nas sintéticas continuam aqui.

    python src/matching.py
"""
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from catalog import CATALOG_DIR, load_stars, load_constellations, constellation_center
from synthetic import gnomonic

TRIANGLES_NPZ = CATALOG_DIR / "triangles.npz"

# triângulos muito achatados são instáveis
MIN_RATIO = 0.05


def triangle_descriptor(p: np.ndarray):
    """(b/a, c/a) com a >= b >= c, e a ordem dos vértices (oposto a a, b, c)
    para saber depois que vértice corresponde a qual."""
    sides = np.array([
        np.linalg.norm(p[1] - p[2]),   # oposto ao vértice 0
        np.linalg.norm(p[0] - p[2]),   # oposto ao vértice 1
        np.linalg.norm(p[0] - p[1]),   # oposto ao vértice 2
    ])
    order = np.argsort(-sides)
    a, b, c = sides[order]
    return np.array([b / a, c / a]), order


def constellation_stars(abbr: str, stars: pd.DataFrame, consts: dict, max_stars: int) -> pd.DataFrame:
    """As estrelas mais brilhantes das linhas da constelação."""
    hips = {h for line in consts[abbr]["lines"] for h in line}
    sel = stars[stars["hip"].isin(hips)].drop_duplicates("hip")
    return sel.nsmallest(max_stars, "mag")      # magnitude menor = mais brilhante


def build_catalog_index(max_stars: int = 10):
    """Triângulos das 10 estrelas mais brilhantes de cada constelação, no plano tangente."""
    stars, consts = load_stars(), load_constellations()
    descs, hips, cons = [], [], []

    for abbr in consts:
        sel = constellation_stars(abbr, stars, consts, max_stars)
        if len(sel) < 3:
            continue
        ra0, dec0 = constellation_center(abbr, stars, consts)
        xi, eta, _ = gnomonic(sel["ra_deg"].values, sel["dec_deg"].values, ra0, dec0)
        pts = np.column_stack([xi, eta])
        hip_arr = sel["hip"].values.astype(int)

        for tri in combinations(range(len(sel)), 3):
            tri = np.array(tri)
            desc, order = triangle_descriptor(pts[tri])
            if desc[1] < MIN_RATIO:
                continue
            descs.append(desc)
            hips.append(hip_arr[tri[order]])
            cons.append(abbr)

    index = {"desc": np.array(descs), "hips": np.array(hips), "con": np.array(cons)}
    np.savez(TRIANGLES_NPZ, **index)
    return index


def load_catalog_index():
    """Carrega (ou calcula) os triângulos e monta a KD-tree."""
    if not TRIANGLES_NPZ.exists():
        build_catalog_index()
    data = np.load(TRIANGLES_NPZ)
    index = {k: data[k] for k in data.files}
    index["tree"] = cKDTree(index["desc"])
    index["planes"] = constellation_planes()
    return index


def constellation_planes() -> dict:
    """Todas as estrelas das linhas de cada constelação no plano tangente
    (mesmo centro que build_catalog_index)."""
    stars, consts = load_stars(), load_constellations()
    planes = {}
    for abbr in consts:
        hips = {h for line in consts[abbr]["lines"] for h in line}
        sel = stars[stars["hip"].isin(hips)].drop_duplicates("hip")
        ra0, dec0 = constellation_center(abbr, stars, consts)
        xi, eta, _ = gnomonic(sel["ra_deg"].values, sel["dec_deg"].values, ra0, dec0)
        planes[abbr] = {"hip": sel["hip"].values.astype(int), "pts": np.column_stack([xi, eta]),
                        "center": (ra0, dec0)}
    return planes


def image_triangles(det: pd.DataFrame, n_stars: int = 20):
    """Triângulos das n_stars deteções mais brilhantes (20 -> 1140 triângulos)."""
    pts = det[["x", "y"]].values[:n_stars]
    descs, ids = [], []
    for tri in combinations(range(len(pts)), 3):
        tri = np.array(tri)
        desc, order = triangle_descriptor(pts[tri])
        if desc[1] < MIN_RATIO:
            continue
        descs.append(desc)
        ids.append(tri[order])
    return np.array(descs).reshape(-1, 2), np.array(ids).reshape(-1, 3)


def find_candidates(img_desc: np.ndarray, index: dict, eps: float = 0.01):
    """Pares (triângulo da imagem, triângulo do catálogo) com descritor a menos de eps."""
    neighbors = index["tree"].query_ball_point(img_desc, r=eps)
    return [(i, j) for i, js in enumerate(neighbors) for j in js]


def fit_similarity(src: np.ndarray, dst: np.ndarray):
    """Semelhança src -> dst por mínimos quadrados (Umeyama), sem espelho.
    dst ≈ scale * R @ src + t"""
    mu_s, mu_d = src.mean(0), dst.mean(0)
    S, D = src - mu_s, dst - mu_d
    U, sig, Vt = np.linalg.svd(D.T @ S)
    d = np.sign(np.linalg.det(U @ Vt))          # sem espelho
    R = U @ np.diag([1, d]) @ Vt
    scale = (sig * [1, d]).sum() / (S ** 2).sum()
    t = mu_d - scale * R @ mu_s
    return scale, R, t


def verify(cand_i, cand_j, img_ids, det_xy, det_tree, index, tol_px: float, img_shape):
    """Teste rápido de um candidato: põe a constelação na imagem com a semelhança
    e conta quantas estrelas batem. Só serve para ordenar hipóteses."""
    abbr = index["con"][cand_j]
    plane = index["planes"][abbr]
    hip_to_k = {h: k for k, h in enumerate(plane["hip"])}
    src = plane["pts"][[hip_to_k[h] for h in index["hips"][cand_j]]]
    dst = det_xy[img_ids[cand_i]]

    scale, R, t = fit_similarity(src, dst)
    h, w = img_shape
    if scale < 0.05 * w:            # demasiado pequena
        return None

    for _ in range(2):
        proj = scale * plane["pts"] @ R.T + t
        dist, k = det_tree.query(proj)
        ok = dist < tol_px
        if ok.sum() < 3:
            return None
        scale, R, t = fit_similarity(plane["pts"][ok], det_xy[k[ok]])

    matches = {int(k[m]): int(plane["hip"][m]) for m in np.where(ok)[0]}
    return {"abbr": abbr, "n": len(matches), "matches": matches,
            "scale": scale, "angle": np.degrees(np.arctan2(R[1, 0], R[0, 0])),
            "T_inv": lambda p, s=scale, R=R, t=t: (p - t) @ R / s}


def identify(det: pd.DataFrame, index: dict, img_shape, eps: float = 0.01,
             n_stars: int = 20, tol_frac: float = 0.006, n_hyp: int = 300,
             n_score: int = 30, min_score: float = 0.5, min_con_stars: int = 4,
             max_fov: float = 100.0, verify_mult: float = 1.0,
             order: str = "both", stop_score: float = 0.9):
    """Versão no plano (antiga). Devolve {} se não houver solução, senão
    pose, score, stars e constellations."""
    from catalog import load_stars
    import projection as P

    det_xy = det[["x", "y"]].values
    det_tree = cKDTree(det_xy)
    tol_px = tol_frac * img_shape[1]
    catalog = index.setdefault("stars", load_stars())
    hip_row = index.setdefault("hip_row", pd.Series(np.arange(len(catalog)), index=catalog["hip"].fillna(-1).astype(int)))

    # teste rápido a todos os candidatos
    desc, ids = image_triangles(det, n_stars)
    if "tri_maxmag" not in index:
        mag = pd.Series(catalog["mag"].values, index=catalog["hip"].fillna(-1).astype(int))
        mag = mag[~mag.index.duplicated()]
        index["tri_maxmag"] = mag.reindex(index["hips"].ravel()).values.reshape(-1, 3).max(1)
    hyps = []
    for i, j in find_candidates(desc, index, eps):
        hy = verify(i, j, ids, det_xy, det_tree, index, verify_mult * tol_px, img_shape)
        if hy:
            hy["faint_det"] = ids[i].max()
            hy["faint_cat"] = index["tri_maxmag"][j]
            hyps.append(hy)
    # brilho funcionava melhor nas reais, inliers nas sintéticas -> alternar
    by_bright = sorted(hyps, key=lambda hy: (hy["faint_det"], hy["faint_cat"], -hy["n"]))
    by_inliers = sorted(hyps, key=lambda hy: -hy["n"])
    if order == "brightness":
        hyps = by_bright
    elif order == "inliers":
        hyps = by_inliers
    else:
        hyps = [hy for pair in zip(by_bright, by_inliers) for hy in pair]

    # pose de cada hipótese, pontuada com o catálogo inteiro
    best, seen = None, set()
    n_top = min(n_score, len(det))
    for hy in hyps:
        key = (hy["abbr"], frozenset(hy["matches"].items()))
        if key in seen:
            continue
        seen.add(key)
        if len(seen) > n_hyp:
            break
        center = index["planes"][hy["abbr"]]["center"]
        pose = P.initial_pose(hy["T_inv"], hy["angle"], hy["scale"], center, img_shape)
        k = np.array(list(hy["matches"]))
        rows = hip_row[list(hy["matches"].values())].values
        try:
            pose = P.refine_pose(pose, catalog.iloc[rows], det_xy[k], img_shape)
        except ValueError:
            continue
        if not (1.0 < pose[3] < max_fov):
            continue
        di, _ = P.match_catalog(pose, catalog, det_xy[:n_top], img_shape, tol_px,
                                n_brightest=3 * n_top)
        score = len(di) / n_top
        if best is None or score > best[0]:
            best = (score, pose)
        if score >= stop_score:
            break

    if best is None or best[0] < min_score:
        return {}

    score, pose = best
    return finalize(pose, score, det_xy, index, img_shape, tol_px, min_con_stars)


FINAL_SCHEDULES = {
    # (x tolerância, catálogo, nº de deteções)
    "simples": ((1, "all", None), (1, "all", None)),
    "grosso_fino": ((4, "bright", 60), (2, "bright", 60), (1, "all", None), (1, "all", None)),
    "suave": ((2, "bright", None), (1, "all", None), (1, "all", None)),
    "suave_k1": ((2, "bright_k1", None), (1, "all", None), (1, "all", None)),
    "brilhantes_k1": ((3, "bright_k1", 60), (2, "bright_k1", 60), (1, "all", None), (1, "all", None)),
}


def finalize(pose, score, det_xy, index, img_shape, tol_px, min_con_stars=4, fit_lens: bool = False,
             schedule: str = "suave", final_div: float = 1.0):
    """Com uma pose boa: emparelha tudo, refina e escolhe as constelações a reportar."""
    import projection as P
    catalog = index["stars"]
    # primeiro só as brilhantes com tolerância larga, depois tudo
    bright = catalog[catalog["mag"] <= 4.5]
    for mult, which, n_det in FINAL_SCHEDULES[schedule]:
        cat = bright if which.startswith("bright") else catalog
        di, ci = P.match_catalog(pose, cat, det_xy[:n_det], img_shape, mult * tol_px)
        if len(di) >= 3:
            # fit_lens desligado: fazia overfitting
            if which == "bright_k1" and len(di) >= 8:
                ext = "k1"
            elif len(pose) == 7:
                ext = "k1"
            else:
                ext = fit_lens and n_det is None and len(di) >= 20
            pose = P.refine_pose(pose, cat.iloc[ci], det_xy[di], img_shape, extended=ext)
    di, ci = P.match_catalog(pose, catalog, det_xy, img_shape, tol_px / final_div)

    stars = pd.DataFrame({"det": di, "hip": catalog["hip"].fillna(-1).values[ci].astype(int),
                          "proper": catalog["proper"].values[ci], "x": det_xy[di, 0], "y": det_xy[di, 1]})
    found = set(stars["hip"])
    _, inside = P.project(catalog, pose, img_shape)
    in_view = set(catalog["hip"].fillna(-1).astype(int).values[inside])
    consts = {}
    for a, p in index["planes"].items():
        hs = [h for h in p["hip"] if h in found]
        n_view = sum(h in in_view for h in p["hip"])
        if report_constellation(len(hs), n_view, min_con_stars):
            consts[a] = hs
    return {"pose": pose, "score": score, "stars": stars, "constellations": consts}


def report_constellation(n_found: int, n_in_view: int, min_con_stars: int = 4) -> bool:
    """>= 4 estrelas, ou >= 3 que sejam 60% das visíveis (para o Cruzeiro do Sul)."""
    return n_found >= min_con_stars or (n_found >= 3 and n_found >= 0.6 * n_in_view)


def _truth_hips(det: pd.DataFrame, truth: pd.DataFrame, max_dist: float = 3.0) -> np.ndarray:
    """HIP verdadeiro de cada deteção (só para avaliar)."""
    d, j = cKDTree(truth[["x", "y"]].values).query(det[["x", "y"]].values)
    hips = truth["hip"].values[j].astype(int)
    return np.where(d < max_dist, hips, -1)


def evaluate_candidates(eps: float = 0.01, n_stars: int = 20):
    """Quantos candidatos e quantos certos por imagem sintética."""
    import cv2
    from preprocessing import preprocess
    from detection import detect_stars

    index = load_catalog_index()
    synth = Path(__file__).resolve().parent.parent / "data" / "synthetic"
    rows = []
    for png in sorted(synth.glob("sky_*.png")):
        gray = cv2.imread(str(png), cv2.IMREAD_GRAYSCALE)
        det = detect_stars(preprocess(gray))
        truth = pd.read_csv(png.with_name(png.stem + "_truth.csv"))
        det_hip = _truth_hips(det, truth)

        desc, ids = image_triangles(det, n_stars)
        cands = find_candidates(desc, index, eps)
        right = sum(np.array_equal(det_hip[ids[i]], index["hips"][j]) for i, j in cands)
        rows.append({"imagem": png.stem, "triangulos": len(desc),
                     "candidatos": len(cands), "certos": right})

    df = pd.DataFrame(rows)
    print(df.to_string(index=False))
    ok = (df["certos"] > 0).mean()
    prec = df["certos"].sum() / max(df["candidatos"].sum(), 1)
    print(f"\neps={eps}, n_stars={n_stars}: imagens com pelo menos 1 candidato certo: {ok:.0%}; "
          f"candidatos certos: {prec:.2%}")
    return df


def evaluate_identification(min_con_stars: int = 4, identify_fn=None, index=None, **kw):
    """TP/FP/FN das constelações, estrelas certas e erro da pose nas sintéticas."""
    import time
    import cv2
    from detection import find_stars

    index = index or load_catalog_index()
    synth = Path(__file__).resolve().parent.parent / "data" / "synthetic"
    rows = []
    for png in sorted(synth.glob("sky_*.png")):
        gray = cv2.imread(str(png), cv2.IMREAD_GRAYSCALE)
        det = find_stars(gray)
        det_hip = _truth_hips(det, pd.read_csv(png.with_name(png.stem + "_truth.csv")))
        params = dict(l.split("=") for l in png.with_name(png.stem + "_params.txt").read_text().split())
        ra_t, dec_t = float(params["ra0"]), float(params["dec0"])

        truth = pd.read_csv(png.with_name(png.stem + "_truth.csv"))
        present = {a for a, p in index["planes"].items()
                   if report_constellation(np.isin(p["hip"], det_hip).sum(),
                                           np.isin(p["hip"], truth["hip"]).sum(), min_con_stars)}
        t0 = time.perf_counter()
        res = (identify_fn or identify)(det, index, gray.shape, min_con_stars=min_con_stars, **kw)
        dt = time.perf_counter() - t0
        found = set(res.get("constellations", {}))

        if res:
            st = res["stars"]
            good, n = int((det_hip[st["det"].values] == st["hip"].values).sum()), len(st)
            ra, dec = res["pose"][:2]
            cosd = np.sin(np.radians(dec)) * np.sin(np.radians(dec_t)) + \
                np.cos(np.radians(dec)) * np.cos(np.radians(dec_t)) * np.cos(np.radians(ra - ra_t))
            err = np.degrees(np.arccos(np.clip(cosd, -1, 1)))
        else:
            good, n, err = 0, 0, np.nan
        rows.append({"imagem": png.stem, "presentes": len(present),
                     "TP": len(present & found), "FP": len(found - present), "FN": len(present - found),
                     "estrelas": n, "certas": good, "erro_centro_graus": round(err, 3),
                     "score": round(res.get("score", 0), 2), "seg": round(dt, 1)})

    df = pd.DataFrame(rows)
    print(df.to_string(index=False))
    tp, fp, fn = df["TP"].sum(), df["FP"].sum(), df["FN"].sum()
    print(f"\nconstelações: precisão {tp / max(tp + fp, 1):.0%}, recall {tp / max(tp + fn, 1):.0%} | "
          f"estrelas com HIP certo: {df['certas'].sum() / max(df['estrelas'].sum(), 1):.1%} | "
          f"imagens resolvidas: {(df['estrelas'] > 0).mean():.0%} | tempo médio {df['seg'].mean():.1f} s")
    return df


if __name__ == "__main__":
    idx = build_catalog_index()
    n = len(idx["desc"])
    print(f"{n} triângulos de {len(set(idx['con']))} constelações guardados em {TRIANGLES_NPZ.name}")

    # o descritor tem de dar igual nos três
    rng = np.random.default_rng(0)
    p = rng.random((3, 2))
    ang = 1.0
    R = np.array([[np.cos(ang), -np.sin(ang)], [np.sin(ang), np.cos(ang)]])
    q = 3.7 * p @ R.T + [5, -2]
    m = p * [-1, 1]
    print("invariância:", triangle_descriptor(p)[0], triangle_descriptor(q)[0], triangle_descriptor(m)[0])

    tree = cKDTree(idx["desc"])
    d, _ = tree.query(idx["desc"], k=2)
    nn = d[:, 1]
    print(f"distância ao triângulo mais parecido: mediana {np.median(nn):.4f}, "
          f"{(nn < 0.01).mean():.0%} têm outro a menos de 0.01")

    print()
    evaluate_candidates(eps=0.01, n_stars=20)

    print()
    evaluate_identification()
