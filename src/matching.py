"""
Fase 3: identificar as estrelas detetadas (matching com o catálogo).

Ideia: um triângulo de 3 estrelas tem uma forma que não muda quando a foto
roda, se desloca ou muda de escala (zoom). Ordenando os lados a >= b >= c,
o par (b/a, c/a) é essa "impressão digital" da forma. Comparamos as
impressões digitais dos triângulos da imagem com as do catálogo.

Passo 1 (este ficheiro, por agora): calcular os descritores de todos os
triângulos do catálogo e guardá-los numa árvore de pesquisa (KD-tree).

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
    return index


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
