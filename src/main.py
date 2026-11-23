"""
Identificar constelações numa fotografia do céu.

    python src/main.py foto.jpg [outra.jpg ...] [--out pasta] [--mask 2]
    python src/main.py --samples
"""
import argparse
import time
from pathlib import Path

import cv2

from catalog import load_stars, load_constellations
from detection import find_stars
from matching_sphere import load_sphere_index, identify_sphere
from overlay import draw_result, draw_failure
from segmentation import sky_mask

ROOT = Path(__file__).resolve().parent.parent

# se uma falhar passa à seguinte; olho-de-peixe só no fim
ATTEMPTS = ({"rank": "flux"}, {"rank": "clip"}, {"rank": "clf"},
            {"rank": "flux", "lens": "fish"}, {"rank": "clip", "lens": "fish"})
MANY_DET = 1500         # mais do que isto costuma ser uma foto tremida (estrelas partidas em riscos)


def solve(gray, index, attempts=ATTEMPTS):
    """Deteção + matching. Devolve (resultado, deteções, nº da tentativa); {} se nada resultar.
    Se falhar com deteções a mais, tenta outra vez com a imagem reduzida a 50% e 25%."""
    det = None
    for i, kw in enumerate(attempts):
        det = find_stars(gray, rank=kw["rank"])
        result = identify_sphere(det, index, gray.shape, lens=kw.get("lens", "rect"))
        if result:
            return result, det, i
    if len(det) > MANY_DET:
        for k, f in enumerate((0.5, 0.25)):
            small = cv2.resize(gray, None, fx=f, fy=f, interpolation=cv2.INTER_AREA)
            for kw in attempts[:2]:
                d = find_stars(small, rank=kw["rank"])
                result = identify_sphere(d, index, small.shape, lens=kw.get("lens", "rect"))
                if result:      # a pose não depende da resolução; só as posições mudam
                    result["stars"][["x", "y"]] /= f
                    d[["x", "y"]] /= f
                    return result, d, len(attempts) + k
    return {}, det, None


def run(path: Path, out_dir: Path, index, stars, consts) -> dict:
    img = cv2.imread(str(path))
    if img is None:
        print(f"{path.name}: não consegui abrir a imagem")
        return {}
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    t0 = time.perf_counter()
    result, det, _ = solve(gray, index)
    dt = time.perf_counter() - t0

    if result:
        names = ", ".join(consts[a]["name"] for a in sorted(result["constellations"])) or "(nenhuma completa)"
        lens = ", olho-de-peixe" if len(result["pose"]) == 5 else ""
        print(f"{path.name}: {names}  [FOV {result['pose'][3]:.0f}°{lens}, score {result['score']:.2f}, {dt:.1f} s]")
        out = draw_result(img, result, stars, consts, sky=sky_mask(gray))
    else:
        print(f"{path.name}: sem solução ({len(det)} deteções, {dt:.1f} s)")
        out = draw_failure(img)

    out_dir.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_dir / f"{path.stem}_resultado.jpg"), out, [cv2.IMWRITE_JPEG_QUALITY, 90])
    return result


def main():
    ap = argparse.ArgumentParser(description="Identificar constelações em fotografias do céu.")
    ap.add_argument("images", nargs="*", type=Path)
    ap.add_argument("--samples", action="store_true", help="correr em todas as fotos de data/samples/")
    ap.add_argument("--out", type=Path, default=ROOT / "results" / "fotos_identificadas")
    ap.add_argument("--mask", type=int, choices=[1, 2], default=1,
                    help="versão da máscara do céu (ver segmentation.py)")
    args = ap.parse_args()
    import segmentation
    segmentation.DEFAULT_VERSION = args.mask

    paths = list(args.images)
    if args.samples:
        paths += sorted((ROOT / "data" / "samples").glob("*.jpg"))
    if not paths:
        ap.error("indica pelo menos uma imagem, ou --samples")

    index, stars, consts = load_sphere_index(), load_stars(), load_constellations()
    ok = sum(bool(run(p, args.out, index, stars, consts)) for p in paths)
    print(f"\n{ok}/{len(paths)} resolvidas. Resultados em {args.out}")


if __name__ == "__main__":
    main()
