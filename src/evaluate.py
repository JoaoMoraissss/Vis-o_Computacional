"""
Avaliação nas fotos reais (e sintéticas com --synthetic).

    python src/evaluate.py --set test4|test3|test2|test|samples|all [--mask 2] [--out pasta]

Cada pasta tem um labels.csv; linha sem constelações = controlo negativo.
Veredictos: certo, errado, falhou, rejeitou. Métricas explicadas no relatório.
"""
import argparse
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

import projection as P
from catalog import load_constellations
from detection import find_stars
from main import solve
from matching_sphere import load_sphere_index, identify_sphere
from segmentation import sky_mask

ROOT = Path(__file__).resolve().parent.parent
SETS = {"samples": ROOT / "data" / "samples", "test": ROOT / "data" / "test",
        "test2": ROOT / "data" / "test2", "test3": ROOT / "data" / "test3", "test4": ROOT / "data" / "test4"}


def load_labels(folder: Path) -> dict:
    lab = pd.read_csv(folder / "labels.csv", keep_default_na=False)
    return {r.ficheiro: set(filter(None, r.esperadas.split(";"))) for r in lab.itertuples()}


def solve_tracked(gray, index):
    return solve(gray, index)


def pose_metrics(res, det, gray, index, tol_px):
    """erro_px, recall e precisão@30 (a pose faz de verdade)."""
    cat = index["stars"]
    h, w = gray.shape
    xy_cat, inside = P.project(cat, res["pose"], gray.shape)

    st = res["stars"]
    row = pd.Series(np.arange(len(cat)), index=cat["hip"].fillna(-1).astype(int).values)
    row = row[~row.index.duplicated()]
    pred = xy_cat[row.reindex(st["hip"]).values]
    erro = float(np.median(np.hypot(*(pred - st[["x", "y"]].values).T))) if len(st) else np.nan

    det_xy = det[["x", "y"]].values
    sky = sky_mask(gray)
    xi = np.clip(xy_cat[:, 0].round().astype(int), 0, w - 1)
    yi = np.clip(xy_cat[:, 1].round().astype(int), 0, h - 1)
    in_sky = inside & sky[yi, xi]

    bright = xy_cat[in_sky & (cat["mag"].values <= 4.5)]
    recall = np.nan
    if len(bright) and len(det_xy):
        d, _ = cKDTree(det_xy).query(bright)
        recall = float((d < tol_px).mean())

    faint_ok = xy_cat[inside & (cat["mag"].values <= 6.5)]
    prec = np.nan
    if len(faint_ok) and len(det_xy):
        d, _ = cKDTree(faint_ok).query(det_xy[:30])
        prec = float((d < tol_px).mean())
    return erro, recall, prec


def evaluate_folder(folder: Path, index, consts) -> pd.DataFrame:
    labels = load_labels(folder)
    rows = []
    for name, expected in labels.items():
        gray = cv2.imread(str(folder / name), cv2.IMREAD_GRAYSCALE)
        if gray is None:
            print(f"{name}: não consegui abrir")
            continue
        t0 = time.perf_counter()
        res, det, attempt = solve_tracked(gray, index)
        dt = time.perf_counter() - t0

        found = set(res.get("constellations", {}))
        if not expected:
            verdict = "errado" if res else "rejeitou"
        elif not res:
            verdict = "falhou"
        else:
            verdict = "certo" if found & expected else "errado"

        r = {"foto": name, "esperadas": ",".join(sorted(expected)) or "-",
             "encontradas": ",".join(sorted(found)) or "-", "veredicto": verdict,
             "tentativa": "-" if attempt is None else attempt + 1,
             "estrelas": len(res["stars"]) if res else 0,
             "score": round(res.get("score", 0), 2), "seg": round(dt, 1),
             "fov": round(float(res["pose"][3]), 1) if res else np.nan,
             "erro_px": np.nan, "det_recall": np.nan, "det_prec30": np.nan}
        if res:
            tol = 0.006 * max(gray.shape)
            e, rc, pr = pose_metrics(res, det, gray, index, tol)
            r.update(erro_px=round(e, 1), det_recall=round(rc, 2), det_prec30=round(pr, 2))
        rows.append(r)
        print(f"  {name:32s} {verdict:8s} {r['encontradas']:22s} {dt:5.1f} s")
    return pd.DataFrame(rows)


def to_markdown(df: pd.DataFrame) -> str:
    """Para não depender do tabulate."""
    cells = [[str(v) for v in df.columns]] + [["" if pd.isna(v) else str(v) for v in r] for r in df.values]
    lines = ["| " + " | ".join(c) + " |" for c in cells]
    lines.insert(1, "|" + "---|" * len(df.columns))
    return "\n".join(lines)


def summary(df: pd.DataFrame, title: str) -> str:
    pos = df[df["esperadas"] != "-"]
    neg = df[df["esperadas"] == "-"]
    certo = (pos["veredicto"] == "certo").sum()
    errado = (pos["veredicto"] == "errado").sum()
    falhou = (pos["veredicto"] == "falhou").sum()
    ok = df[df["veredicto"] == "certo"]
    lines = [f"## {title}", "",
             f"- fotos com constelações: {len(pos)} -> certas {certo} ({certo / max(len(pos), 1):.0%}), "
             f"erradas {errado}, sem solução {falhou}"]
    if len(neg):
        lines.append(f"- controlos negativos: {len(neg)} -> corretamente rejeitados "
                     f"{(neg['veredicto'] == 'rejeitou').sum()}")
    solved = df[df["veredicto"].isin(["certo", "errado"])]
    if len(solved):
        lines.append(f"- precisão (das soluções dadas, quantas certas): {len(ok) / len(solved):.0%}")
    if len(ok):
        lines.append(f"- nas certas: erro da pose mediano {ok['erro_px'].median():.1f} px, "
                     f"recall da deteção (mag<=4.5) {ok['det_recall'].mean():.0%}, "
                     f"precisão das 30 mais brilhantes {ok['det_prec30'].mean():.0%}, "
                     f"{ok['estrelas'].mean():.0f} estrelas identificadas por foto")
    lines.append(f"- tempo médio {df['seg'].mean():.1f} s/foto; resolvidas à 1.ª tentativa: "
                 f"{(df['tentativa'].astype(str) == '1').sum()}/{len(solved)}")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description="Avaliar o pipeline em fotografias reais.")
    ap.add_argument("--set", choices=["test", "test2", "test3", "test4", "samples", "all"], default="test4")
    ap.add_argument("--synthetic", action="store_true", help="avaliar também as imagens sintéticas")
    ap.add_argument("--mask", type=int, choices=[1, 2], default=1,
                    help="versão da máscara do céu (ver segmentation.py)")
    ap.add_argument("--out", type=Path, default=ROOT / "results" / "metrics")
    args = ap.parse_args()
    import segmentation
    segmentation.DEFAULT_VERSION = args.mask

    index, consts = load_sphere_index(), load_constellations()
    args.out.mkdir(parents=True, exist_ok=True)
    names = list(SETS) if args.set == "all" else [args.set]
    titles = {"samples": "Desenvolvimento (data/samples, usadas para afinar)",
              "test": "Teste 1 (data/test; usado para diagnosticar a máscara do céu)",
              "test2": "Teste 2 (data/test2; usado para corrigir a máscara do céu)",
              "test3": "Teste 3 (data/test3)",
              "test4": "Teste 4 (data/test4, bloqueado: nunca visto)"}
    report = ["# Resultados da avaliação", ""]
    for s in names:
        print(f"\n{titles[s]}")
        df = evaluate_folder(SETS[s], index, consts)
        df.to_csv(args.out / f"eval_{s}.csv", index=False)
        report += [summary(df, titles[s]), "", to_markdown(df), ""]

    if args.synthetic:
        import matching as M
        print("\nSintéticas")
        sdf = M.evaluate_identification(identify_fn=identify_sphere, index=index)
        sdf.to_csv(args.out / "eval_synthetic.csv", index=False)

    text = "\n".join(report)
    (args.out / "resultados.md").write_text(text)
    print("\n" + "\n\n".join(summary(pd.read_csv(args.out / f"eval_{s}.csv"), titles[s]) for s in names))
    print(f"\nTabelas em {args.out}")


if __name__ == "__main__":
    main()
