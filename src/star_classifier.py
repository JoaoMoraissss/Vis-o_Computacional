"""
Classificador estrela / artefacto para cada deteção (regressão logística em numpy).

As etiquetas saem de graça das fotos já resolvidas: a pose diz que deteções caem em
cima de uma estrela do catálogo (1) e quais não (0: luzes, ramos, nós de nebulosa).
Treina-se só com fotos de desenvolvimento; o teste 4 não entra.

    python src/star_classifier.py            # cria o conjunto de treino, valida e grava o modelo
"""
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
MODEL = ROOT / "data" / "catalog" / "star_classifier.json"
TRAIN_SETS = ("samples", "test", "test2", "test3")      # já não são limpos
N_TOP = 60          # só as mais brilhantes: as fracas podem ser estrelas reais fora do catálogo

FEATURES = ["log_flux", "log_peak", "log_area", "elong", "escala", "concentracao",
            "fundo", "textura", "y_rel"]


def features(det: pd.DataFrame, gray: np.ndarray) -> np.ndarray:
    h, w = gray.shape
    r0 = max(3, round(0.004 * w))
    g = gray.astype(np.float32)
    med, mad = np.median(g), np.median(np.abs(g - np.median(g))) + 1
    small = cv2.resize(g, (w // 8, h // 8), interpolation=cv2.INTER_AREA)
    bg = cv2.medianBlur(small.astype(np.uint8), 5).astype(np.float32)
    m = cv2.blur(bg, (7, 7))
    tex = np.sqrt(np.maximum(cv2.blur(bg * bg, (7, 7)) - m * m, 0))
    xi = (det["x"].values / 8).astype(int).clip(0, bg.shape[1] - 1)
    yi = (det["y"].values / 8).astype(int).clip(0, bg.shape[0] - 1)
    f = np.column_stack([
        np.log1p(det["flux"].values), np.log1p(det["peak"].values), np.log(det["area"].values),
        det["elong"].values, np.log2(det["r"].values / r0),
        det["peak"].values * det["area"].values / np.maximum(det["flux"].values, 1e-6),
        (bg[yi, xi] - med) / mad, tex[yi, xi] / mad, det["y"].values / h])
    f = np.nan_to_num(f)
    # posição relativa dentro da foto (0 = menor, 1 = maior): o brilho absoluto
    # varia muito de foto para foto, a ordem não
    return f.argsort(0).argsort(0) / max(len(f) - 1, 1)


class LogReg:
    """Regressão logística com regularização L2, treinada por descida de gradiente."""

    def __init__(self, lam=1e-2):
        self.lam = lam

    def fit(self, X, y, iters=3000, lr=0.5):
        self.mu, self.sd = X.mean(0), X.std(0) + 1e-9
        Z = np.column_stack([np.ones(len(X)), (X - self.mu) / self.sd])
        wpos = len(y) / (2 * max(y.sum(), 1)); wneg = len(y) / (2 * max(len(y) - y.sum(), 1))
        sw = np.where(y == 1, wpos, wneg)             # classes equilibradas
        self.w = np.zeros(Z.shape[1])
        for _ in range(iters):
            p = 1 / (1 + np.exp(-Z @ self.w))
            grad = Z.T @ (sw * (p - y)) / len(y) + self.lam * np.r_[0, self.w[1:]]
            self.w -= lr * grad
        return self

    def predict_proba(self, X):
        Z = np.column_stack([np.ones(len(X)), (X - self.mu) / self.sd])
        return 1 / (1 + np.exp(-Z @ self.w))

    def save(self, path):
        path.write_text(json.dumps({"features": FEATURES, "mu": self.mu.tolist(),
                                    "sd": self.sd.tolist(), "w": self.w.tolist()}, indent=1))

    @classmethod
    def load(cls, path=MODEL):
        d = json.loads(Path(path).read_text())
        m = cls(); m.mu, m.sd, m.w = np.array(d["mu"]), np.array(d["sd"]), np.array(d["w"])
        return m


def star_labels(res, det, gray, index):
    """1 se a deteção cai a menos da tolerância de uma estrela do catálogo (até mag 6,5)
    projetada com a pose encontrada."""
    import projection as P
    from scipy.spatial import cKDTree
    cat = index["stars"]
    xy, inside = P.project(cat, res["pose"], gray.shape)
    xy = xy[inside & (cat["mag"].values <= 6.5)]
    d, _ = cKDTree(xy).query(det[["x", "y"]].values)
    return (d < 0.006 * max(gray.shape)).astype(float)


def build_dataset():
    """Deteções etiquetadas das fotos resolvidas e certas dos conjuntos de treino."""
    from evaluate import SETS, load_labels, solve_tracked
    from matching_sphere import load_sphere_index
    index = load_sphere_index()
    X, y, grupo = [], [], []
    for s in TRAIN_SETS:
        for name, expected in load_labels(SETS[s]).items():
            if not expected:
                continue
            gray = cv2.imread(str(SETS[s] / name), cv2.IMREAD_GRAYSCALE)
            res, det, _ = solve_tracked(gray, index)
            if not res or not (set(res["constellations"]) & expected):
                continue
            top = det.iloc[:N_TOP]
            lab = star_labels(res, top, gray, index)
            X.append(features(top, gray)); y.append(lab); grupo += [f"{s}/{name}"] * len(top)
            print(f"  {s}/{name}: {int(lab.sum())}/{len(top)} estrelas")
    return np.vstack(X), np.concatenate(y), np.array(grupo)


def auc(y, p):
    order = np.argsort(p); r = np.empty(len(p)); r[order] = np.arange(1, len(p) + 1)
    n1 = y.sum(); n0 = len(y) - n1
    return (r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


if __name__ == "__main__":
    X, y, g = build_dataset()
    print(f"\n{len(y)} deteções de {len(set(g))} fotos, {y.mean():.0%} estrelas")
    # validação deixando uma foto de fora de cada vez
    p = np.zeros(len(y))
    for foto in set(g):
        tr = g != foto
        p[~tr] = LogReg().fit(X[tr], y[tr]).predict_proba(X[~tr])
    pred = p >= 0.5
    print(f"validação por foto: AUC {auc(y, p):.3f}, exatidão {(pred == y).mean():.1%}, "
          f"precisão {y[pred].mean():.1%}, recall {pred[y == 1].mean():.1%}")
    m = LogReg().fit(X, y); m.save(MODEL)
    print("pesos:", {f: round(w, 2) for f, w in zip(FEATURES, m.w[1:])})
    np.savez(ROOT / "results" / "metrics" / "classificador_treino.npz", X=X, y=y, g=g, p_cv=p)
