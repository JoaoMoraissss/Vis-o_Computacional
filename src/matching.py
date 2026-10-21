"""
Fase 3: identificar as estrelas detetadas (matching com o catálogo).

Ideia: um triângulo de 3 estrelas tem uma forma que não muda quando a foto
roda, se desloca ou muda de escala (zoom). Ordenando os lados a >= b >= c,
o par (b/a, c/a) é essa "impressão digital" da forma. Comparamos as
impressões digitais dos triângulos da imagem com as do catálogo.

Passo 1: descritores de todos os triângulos do catálogo, numa KD-tree.
Passo 2: descritores dos triângulos das estrelas detetadas.
Passo 3: candidatos = triângulos do catálogo com forma parecida (pesquisa na KD-tree).
Passo 4: verificação. Cada candidato dá uma hipótese de pose da câmara; fica a
         pose que explica mais estrelas detetadas (ver projection.py).

Função principal: identify(det, index, img_shape).

Correr a partir da raiz do projeto:
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

# Triângulos com um lado muito curto em relação ao maior são instáveis:
# um erro de 1 píxel muda muito o rácio c/a. Ficam de fora.
MIN_RATIO = 0.05


def triangle_descriptor(p: np.ndarray):
    """Descritor invariante de um triângulo.

    p: array (3, 2) com as posições dos 3 vértices (em qualquer unidade).
    Devolve:
      desc  -> (b/a, c/a), com a >= b >= c os lados
      order -> índices dos vértices ordenados como "oposto a a", "oposto a b", "oposto a c".
               Esta ordem é a mesma na imagem e no catálogo, por isso diz-nos que
               vértice da imagem corresponde a que vértice do catálogo (precisamos no passo 4).
    """
    # o lado oposto ao vértice i é o que liga os outros dois
    sides = np.array([
        np.linalg.norm(p[1] - p[2]),   # oposto ao vértice 0
        np.linalg.norm(p[0] - p[2]),   # oposto ao vértice 1
        np.linalg.norm(p[0] - p[1]),   # oposto ao vértice 2
    ])
    order = np.argsort(-sides)          # do lado maior para o menor
    a, b, c = sides[order]
    return np.array([b / a, c / a]), order


def constellation_stars(abbr: str, stars: pd.DataFrame, consts: dict, max_stars: int) -> pd.DataFrame:
    """As max_stars estrelas mais brilhantes que fazem parte das linhas da constelação."""
    hips = {h for line in consts[abbr]["lines"] for h in line}
    sel = stars[stars["hip"].isin(hips)].drop_duplicates("hip")
    return sel.nsmallest(max_stars, "mag")      # magnitude menor = mais brilhante


def build_catalog_index(max_stars: int = 10):
    """Calcula os descritores de todos os triângulos de cada constelação.

    Para cada constelação:
      1. escolhe as estrelas mais brilhantes das suas linhas
         (são as que uma câmara apanha primeiro);
      2. projeta-as num plano tangente centrado na constelação (gnomónica,
         a mesma projeção de uma câmara), para as distâncias serem como numa foto;
      3. calcula o descritor de cada combinação de 3 estrelas.
    """
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
            hips.append(hip_arr[tri[order]])     # HIPs já na ordem "oposto a a, b, c"
            cons.append(abbr)

    index = {"desc": np.array(descs), "hips": np.array(hips), "con": np.array(cons)}
    np.savez(TRIANGLES_NPZ, **index)
    return index


def load_catalog_index():
    """Carrega os triângulos do catálogo (calcula-os se ainda não existirem) e a KD-tree."""
    if not TRIANGLES_NPZ.exists():
        build_catalog_index()
    data = np.load(TRIANGLES_NPZ)
    index = {k: data[k] for k in data.files}
    index["tree"] = cKDTree(index["desc"])
    index["planes"] = constellation_planes()
    return index


def constellation_planes() -> dict:
    """Para o passo 4: posição no plano tangente de TODAS as estrelas das linhas de cada constelação.

    Usa o mesmo centro de projeção que build_catalog_index, para os vértices dos
    triângulos e as restantes estrelas estarem no mesmo sistema de coordenadas.
    Devolve {abbr: {"hip": array, "pts": array (n, 2), "center": (ra0, dec0)}}.
    """
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
    """Passo 2: descritores dos triângulos formados pelas estrelas detetadas.

    det: saída de detect_stars (já ordenada do mais brilhante para o menos).
    Usamos só as n_stars mais brilhantes: são as mais fiáveis (as fracas podem
    ser ruído) e o número de triângulos cresce muito depressa, C(n,3):
    20 estrelas -> 1140 triângulos, 40 -> 9880.

    Devolve:
      desc -> (M, 2) descritores
      ids  -> (M, 3) índices das estrelas em det, na ordem "oposto a a, b, c"
    """
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
    """Passo 3: para cada triângulo da imagem, os triângulos do catálogo com forma parecida.

    Procura na KD-tree todos os descritores do catálogo a menos de eps.
    eps é a tolerância: tem de cobrir o erro da deteção (centróides com ~0.5 px
    de erro) e a deformação da projeção, mas se for grande demais aparecem
    imensos falsos candidatos.

    Devolve uma lista de pares (i_imagem, j_catalogo).
    """
    neighbors = index["tree"].query_ball_point(img_desc, r=eps)
    return [(i, j) for i, js in enumerate(neighbors) for j in js]


def fit_similarity(src: np.ndarray, dst: np.ndarray):
    """Transformação de semelhança (rotação + escala + translação) que leva src a dst,
    pelo método dos mínimos quadrados (Umeyama). Precisa de pelo menos 2 pares de pontos.

    Não permite espelho: com a imagem em píxeis (y para baixo) e o plano tangente
    (xi para leste, eta para norte), uma foto normal do céu é só uma rotação do plano,
    nunca um espelho. Proibir o espelho elimina metade dos falsos candidatos.

    Devolve (scale, R, t), com dst ≈ scale * R @ src + t.
    """
    mu_s, mu_d = src.mean(0), dst.mean(0)
    S, D = src - mu_s, dst - mu_d
    U, sig, Vt = np.linalg.svd(D.T @ S)
    d = np.sign(np.linalg.det(U @ Vt))          # força rotação (det = +1)
    R = U @ np.diag([1, d]) @ Vt
    scale = (sig * [1, d]).sum() / (S ** 2).sum()
    t = mu_d - scale * R @ mu_s
    return scale, R, t


def verify(cand_i, cand_j, img_ids, det_xy, det_tree, index, tol_px: float, img_shape):
    """Passo 4a (teste rápido): um candidato (triângulo i da imagem <-> triângulo j do catálogo).

    1. Calcula a semelhança que leva os 3 vértices do catálogo aos 3 da imagem.
    2. Aplica-a a todas as estrelas da constelação -> onde deviam aparecer na imagem.
    3. Conta quantas têm uma deteção a menos de tol_px, refina com essas e conta de novo.

    É só aproximado (a semelhança ignora a deformação da projeção), por isso serve
    para ORDENAR as hipóteses, não para decidir. Quem decide é o passo 4b.
    Devolve um dicionário com a hipótese, ou None se for rejeitada.
    """
    abbr = index["con"][cand_j]
    plane = index["planes"][abbr]
    hip_to_k = {h: k for k, h in enumerate(plane["hip"])}
    src = plane["pts"][[hip_to_k[h] for h in index["hips"][cand_j]]]
    dst = det_xy[img_ids[cand_i]]

    scale, R, t = fit_similarity(src, dst)
    h, w = img_shape
    if scale < 0.05 * w:            # constelação com menos de 5% da largura: irrealista
        return None

    for _ in range(2):              # 1.ª contagem + 1 refinamento
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
             n_stars: int = 20, tol_frac: float = 0.006, n_hyp: int = 100,
             n_score: int = 30, min_score: float = 0.5, min_con_stars: int = 4):
    """Passos 2 a 4: pose da câmara, estrela de cada deteção e constelações na imagem.

    4a. Teste rápido (verify) a todos os candidatos; ficam as n_hyp melhores hipóteses.
    4b. Para cada hipótese, estima a pose da câmara e projeta o catálogo INTEIRO.
        Score = fração das n_score deteções mais brilhantes que caem em cima de uma
        estrela do catálogo. Uma pose errada explica poucas (só por acaso); a certa
        explica quase todas. Fica a melhor, se o score passar de min_score.
    4c. Com a pose escolhida, associa todas as deteções ao catálogo e refina a pose.
        Uma constelação é reportada se tiver pelo menos min_con_stars estrelas das linhas identificadas.

    tol_frac: tolerância de posição em fração da largura (0.006 * 1280 px ≈ 8 px).
    Devolve um dicionário (vazio se não encontrar solução) com:
      pose (ra0, dec0, roll, fov), score, stars (DataFrame det -> hip), constellations {abbr: [hips]}
    """
    from catalog import load_stars
    import projection as P

    det_xy = det[["x", "y"]].values
    det_tree = cKDTree(det_xy)
    tol_px = tol_frac * img_shape[1]
    catalog = index.setdefault("stars", load_stars())
    hip_row = index.setdefault("hip_row", pd.Series(np.arange(len(catalog)), index=catalog["hip"].fillna(-1).astype(int)))

    # 4a: teste rápido e ordenação
    desc, ids = image_triangles(det, n_stars)
    hyps = [hy for i, j in find_candidates(desc, index, eps)
            if (hy := verify(i, j, ids, det_xy, det_tree, index, tol_px, img_shape))]
    hyps.sort(key=lambda hy: -hy["n"])

    # 4b: pose da câmara para as melhores hipóteses, avaliada com o catálogo inteiro
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
        di, _ = P.match_catalog(pose, catalog, det_xy[:n_top], img_shape, tol_px)
        score = len(di) / n_top
        if best is None or score > best[0]:
            best = (score, pose)
        if score >= 0.9:            # já explica quase tudo: não vale a pena testar mais
            break

    if best is None or best[0] < min_score:
        return {}

    # 4c: associação final e refinamento com todas as deteções
    score, pose = best
    for _ in range(2):
        di, ci = P.match_catalog(pose, catalog, det_xy, img_shape, tol_px)
        pose = P.refine_pose(pose, catalog.iloc[ci], det_xy[di], img_shape)
    di, ci = P.match_catalog(pose, catalog, det_xy, img_shape, tol_px / 2)

    stars = pd.DataFrame({"det": di, "hip": catalog["hip"].fillna(-1).values[ci].astype(int),
                          "proper": catalog["proper"].values[ci], "x": det_xy[di, 0], "y": det_xy[di, 1]})
    found = set(stars["hip"])
    consts = {a: [h for h in p["hip"] if h in found] for a, p in index["planes"].items()}
    consts = {a: hs for a, hs in consts.items() if len(hs) >= min_con_stars}
    return {"pose": pose, "score": score, "stars": stars, "constellations": consts}


def _truth_hips(det: pd.DataFrame, truth: pd.DataFrame, max_dist: float = 3.0) -> np.ndarray:
    """Só para avaliar: que estrela HIP é cada deteção (a da verdade mais próxima, se a < max_dist px)."""
    d, j = cKDTree(truth[["x", "y"]].values).query(det[["x", "y"]].values)
    hips = truth["hip"].values[j].astype(int)
    return np.where(d < max_dist, hips, -1)


def evaluate_candidates(eps: float = 0.01, n_stars: int = 20):
    """Corre os passos 2 e 3 em todas as imagens sintéticas e mede:
      - quantos candidatos aparecem por imagem;
      - quantos estão certos (os 3 HIPs coincidem, pela mesma ordem);
      - em quantas imagens há pelo menos um candidato certo (sem isso, o passo 4 não tem hipótese).
    """
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


def evaluate_identification(min_con_stars: int = 4, **kw):
    """Avalia o passo 4 nas imagens sintéticas.

    "Presentes" = constelações com pelo menos min_con_stars estrelas das linhas detetadas
    (mesmo critério que identify usa para reportar).
    Mede, por imagem: constelações encontradas certas (TP), inventadas (FP), falhadas (FN),
    a percentagem de estrelas com o HIP certo e o erro da pose estimada.
    """
    import time
    import cv2
    from preprocessing import preprocess
    from detection import detect_stars

    index = load_catalog_index()
    synth = Path(__file__).resolve().parent.parent / "data" / "synthetic"
    rows = []
    for png in sorted(synth.glob("sky_*.png")):
        gray = cv2.imread(str(png), cv2.IMREAD_GRAYSCALE)
        det = detect_stars(preprocess(gray))
        det_hip = _truth_hips(det, pd.read_csv(png.with_name(png.stem + "_truth.csv")))
        params = dict(l.split("=") for l in png.with_name(png.stem + "_params.txt").read_text().split())
        ra_t, dec_t = float(params["ra0"]), float(params["dec0"])

        present = {a for a, p in index["planes"].items() if np.isin(p["hip"], det_hip).sum() >= min_con_stars}
        t0 = time.perf_counter()
        res = identify(det, index, gray.shape, min_con_stars=min_con_stars, **kw)
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

    # Verificação 1: o descritor não muda com rotação, escala, translação nem espelho
    rng = np.random.default_rng(0)
    p = rng.random((3, 2))
    ang = 1.0
    R = np.array([[np.cos(ang), -np.sin(ang)], [np.sin(ang), np.cos(ang)]])
    q = 3.7 * p @ R.T + [5, -2]       # roda, aumenta 3.7x e desloca
    m = p * [-1, 1]                    # espelho
    print("invariância:", triangle_descriptor(p)[0], triangle_descriptor(q)[0], triangle_descriptor(m)[0])

    # Verificação 2: quão "únicos" são os triângulos? (distância ao vizinho mais próximo)
    tree = cKDTree(idx["desc"])
    d, _ = tree.query(idx["desc"], k=2)
    nn = d[:, 1]
    print(f"distância ao triângulo mais parecido: mediana {np.median(nn):.4f}, "
          f"{(nn < 0.01).mean():.0%} têm outro a menos de 0.01")

    # Passos 2 e 3 nas imagens sintéticas
    print()
    evaluate_candidates(eps=0.01, n_stars=20)

    # Passo 4 (pose da câmara + identificação) nas imagens sintéticas
    print()
    evaluate_identification()
