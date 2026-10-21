"""
Fase 3, versão 2: triângulos medidos na esfera celeste.

Problema da versão 1 (matching.py): o descritor (b/a, c/a) era medido em píxeis.
Uma câmara é uma projeção gnomónica, que estica as formas longe do centro:
a θ graus do eixo, a direção radial estica 1/cos²θ e a tangencial só 1/cos θ.
A 35° do centro, um triângulo fica ~22% deformado, e a tolerância era de 1%.
Por isso falhava nas fotos de grande angular (por exemplo, todas as da Ursa Maior).

Solução:
  1. Catálogo: lados dos triângulos em ÂNGULOS no céu (não dependem da projeção).
  2. Imagem: para um FOV suposto, cada píxel vira um raio 3-D e os lados também
     viram ângulos, sem deformação. Como o FOV não se conhece, experimenta-se
     uma série (8° a 100°).
  3. Com o FOV suposto, o TAMANHO do triângulo (lado maior em graus) também tem
     de bater certo, o que elimina a maioria dos falsos candidatos.
  4. Com 3 pares estrela <-> raio, a rotação da câmara sai EXATA pelo algoritmo
     de Kabsch (SVD 3x3). A verificação projeta o catálogo com o modelo de
     câmara verdadeiro, sem a aproximação da semelhança no plano.

Função principal: identify_sphere(det, index, img_shape), com a mesma saída de
matching.identify. O índice vem de load_sphere_index().
"""
from itertools import combinations

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

import matching as M
import projection as P
from catalog import CATALOG_DIR, load_stars, load_constellations

SPHERE_NPZ = CATALOG_DIR / "triangles_sphere.npz"
FOV_GUESSES = np.geomspace(8, 100, 20)          # passo de ~14% entre FOVs seguidos


def radec_to_vec(ra_deg, dec_deg) -> np.ndarray:
    """(RA, Dec) em graus -> vetores unitários (n, 3)."""
    ra, dec = np.radians(np.asarray(ra_deg, float)), np.radians(np.asarray(dec_deg, float))
    return np.column_stack([np.cos(dec) * np.cos(ra), np.cos(dec) * np.sin(ra), np.sin(dec)])


def vec_to_radec(v):
    v = v / np.linalg.norm(v)
    return np.degrees(np.arctan2(v[1], v[0])) % 360, np.degrees(np.arcsin(v[2]))


def triangles_from_angles(A: np.ndarray, combos: np.ndarray):
    """Descritores de muitos triângulos de uma vez.

    A:      matriz (n, n) de distâncias angulares entre n estrelas
    combos: (K, 3) índices das 3 estrelas de cada triângulo
    Devolve desc (K, 2) = (b/a, c/a), a (K,) = lado maior em radianos,
    e ids (K, 3) = vértices na ordem "oposto a a, b, c". Tira os triângulos instáveis.
    """
    i, j, k = combos.T
    sides = np.column_stack([A[j, k], A[i, k], A[i, j]])     # lado oposto a cada vértice
    order = np.argsort(-sides, axis=1)
    s = np.take_along_axis(sides, order, axis=1)
    desc = s[:, 1:] / s[:, :1]
    ok = desc[:, 1] >= M.MIN_RATIO
    return desc[ok], s[ok, 0], np.take_along_axis(combos, order, axis=1)[ok]


def angle_matrix(V: np.ndarray) -> np.ndarray:
    return np.arccos(np.clip(V @ V.T, -1, 1))


def build_sphere_index(max_stars: int = 10):
    """Passo 1 (versão esfera): triângulos das estrelas mais brilhantes de cada constelação."""
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
    """Índice da versão 1 (precisamos dos planos e do catálogo para o passo final)
    mais os triângulos esféricos e os vetores unitários de todas as estrelas."""
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
    index["bright_rows"] = by_mag[mag[by_mag] <= bright_mag]          # já ordenadas por brilho
    return index


def pixel_rays(xy: np.ndarray, f: float, w: int, h: int) -> np.ndarray:
    """Píxeis -> raios unitários no referencial da câmara (x direita, y baixo, z em frente)."""
    D = np.column_stack([xy[:, 0] - w / 2, xy[:, 1] - h / 2, np.full(len(xy), f)])
    return D / np.linalg.norm(D, axis=1, keepdims=True)


def kabsch(V: np.ndarray, D: np.ndarray) -> np.ndarray:
    """Rotação R que melhor leva os vetores V (céu) aos D (câmara): D ≈ V @ R.T.
    É o mesmo truque da SVD de fit_similarity, mas em 3-D e sem escala."""
    U, _, Vt = np.linalg.svd(V.T @ D)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    return Vt.T @ np.diag([1, 1, d]) @ U.T


def project_rows(R, f, rows, vec, w, h):
    """Projeta as estrelas `rows` com a câmara (R, f). Devolve xy e máscara de dentro da imagem."""
    Pc = vec[rows] @ R.T
    z = Pc[:, 2]
    with np.errstate(divide="ignore", invalid="ignore"):
        x = w / 2 + f * Pc[:, 0] / z
        y = h / 2 + f * Pc[:, 1] / z
    inside = (z > 0) & (x >= 0) & (x < w) & (y >= 0) & (y < h)
    return np.column_stack([x, y]), inside


def score_camera(R, f, index, top_xy, img_shape, tol_px, n_brightest):
    """Fração das deteções mais brilhantes explicadas pelas n_brightest estrelas do
    catálogo mais brilhantes dentro da imagem. Devolve (score, pares det -> linha do catálogo)."""
    h, w = img_shape
    rows = index["bright_rows"]
    xy, inside = project_rows(R, f, rows, index["vec"], w, h)
    sel = np.where(inside)[0][:n_brightest]
    if len(sel) < 3:
        return 0.0, []
    d, j = cKDTree(xy[sel]).query(top_xy)
    ok = d < tol_px
    pairs = {}
    for di in np.where(ok)[0]:                   # cada estrela do catálogo só conta uma vez
        pairs.setdefault(int(j[di]), int(di))
    return len(pairs) / len(top_xy), [(di, int(rows[sel[k]])) for k, di in pairs.items()]


def refine_camera(pairs, det_xy, index, f, img_shape, span: float = 0.15, steps: int = 31):
    """Ajusta o FOV (via f) e a rotação às estrelas emparelhadas.

    Com o FOV suposto da grelha (até ~13% errado), a rotação de Kabsch acerta
    no triângulo mas desalinha as estrelas da borda. Aqui experimenta-se f em
    ±span e fica o que deixa os pares mais alinhados (Kabsch com todos os pares).
    """
    h, w = img_shape
    di = np.array([p[0] for p in pairs]); V = index["vec"][[p[1] for p in pairs]]
    best = None
    for fs in f * np.linspace(1 - span, 1 + span, steps):
        D = pixel_rays(det_xy[di], fs, w, h)
        R = kabsch(V, D)
        err = (1 - (V @ R.T * D).sum(1)).sum()
        if best is None or err < best[0]:
            best = (err, R, fs)
    return best[1], best[2]


def camera_to_pose(R, f, pairs, det_xy, index, img_shape):
    """Converte (R, f) para (ra0, dec0, roll, fov) do modelo de projection.py e refina.
    O roll procura-se numa grelha (é só 1 número), o resto sai diretamente de R e f."""
    h, w = img_shape
    ra0, dec0 = vec_to_radec(R.T @ np.array([0.0, 0.0, 1.0]))     # para onde aponta o eixo
    fov = np.degrees(2 * np.arctan((w / 2) / f))
    di = np.array([p[0] for p in pairs]); rows = np.array([p[1] for p in pairs])
    cat = index["stars"].iloc[rows]
    errs = [np.abs(P.project(cat, (ra0, dec0, r, fov), img_shape)[0] - det_xy[di]).sum()
            for r in range(0, 360, 5)]
    pose = np.array([ra0, dec0, 5.0 * int(np.argmin(errs)), fov])
    return P.refine_pose(pose, cat, det_xy[di], img_shape)


def identify_sphere(det: pd.DataFrame, index: dict, img_shape, n_stars: int = 20,
                    eps: float = 0.01, size_tol: float = 0.15, tol_frac: float = 0.006,
                    n_score: int = 30, min_score: float = 0.5, stop_score: float = 0.9,
                    max_hyp: int = 5000, min_con_stars: int = 4, fov_guesses=FOV_GUESSES,
                    coarse_mult: float = 3.0, coarse_min: float = 0.15):
    """Passos 2 a 4 com triângulos na esfera. Mesma saída que matching.identify."""
    h, w = img_shape
    det_xy = det[["x", "y"]].values
    n = min(n_stars, len(det))
    if n < 3:
        return {}
    tol_px = tol_frac * w
    top_xy = det_xy[:min(n_score, len(det))]
    combos = np.array(list(combinations(range(n), 3)))

    # 2 + 3: para cada FOV suposto, triângulos da imagem e candidatos (forma E tamanho)
    hyps, rays = [], {}
    for F in fov_guesses:
        f = (w / 2) / np.tan(np.radians(F) / 2)
        D = pixel_rays(det_xy[:n], f, w, h)
        rays[F] = (f, D)
        desc, a, ids = triangles_from_angles(angle_matrix(D), combos)
        for i, js in enumerate(index["s_tree"].query_ball_point(desc, eps)):
            if not js:
                continue
            js = np.array(js)
            js = js[np.abs(a[i] / index["s_a"][js] - 1) < size_tol]
            for j in js:
                hyps.append((ids[i].max(), index["s_maxmag"][j], F, tuple(ids[i]), j))

    # 4: as hipóteses de estrelas mais brilhantes primeiro; rotação por Kabsch.
    #    O FOV da grelha é aproximado, por isso: contagem larga (filtro) ->
    #    refinar FOV e rotação com os pares encontrados -> contagem final rigorosa.
    hyps.sort(key=lambda t: (t[0], t[1]))
    best = None
    nb = 3 * len(top_xy)
    for _, _, F, tri, j in hyps[:max_hyp]:
        f, D = rays[F]
        R = kabsch(index["vec"][index["s_rows"][j]], D[list(tri)])
        coarse, pairs = score_camera(R, f, index, top_xy, img_shape, coarse_mult * tol_px, nb)
        if coarse < coarse_min:
            continue
        for _ in range(2):
            R, f = refine_camera(pairs, det_xy, index, f, img_shape)
            score, pairs = score_camera(R, f, index, top_xy, img_shape, tol_px, nb)
            if len(pairs) < 3:
                break
        if best is None or score > best[0]:
            best = (score, R, f, pairs)
        if score >= stop_score:
            break

    if best is None or best[0] < min_score:
        return {}
    score, R, f, pairs = best
    pose = camera_to_pose(R, f, pairs, det_xy, index, img_shape)
    if not (1.0 < pose[3] < 120):
        return {}
    return M.finalize(pose, score, det_xy, index, img_shape, tol_px, min_con_stars)


if __name__ == "__main__":
    build_sphere_index()
    idx = load_sphere_index()
    print(f"{len(idx['s_desc'])} triângulos esféricos guardados em {SPHERE_NPZ.name}")
    print()
    M.evaluate_identification(identify_fn=identify_sphere, index=idx)
