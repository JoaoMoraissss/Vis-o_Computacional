"""
Identificar constelações numa fotografia do céu.

Uso (a partir da raiz do projeto):
    python src/main.py foto.jpg [outra.jpg ...]     # guarda em results/demo/
    python src/main.py --samples                    # todas as fotos de data/samples/
    python src/main.py foto.jpg --out pasta/

Pipeline:
    fase 2  find_stars    top-hat + deteção + filtro de forma + máscara do céu
    fase 3  identify      triângulos -> candidatos -> pose da câmara
    saída   draw_result   linhas e nomes das constelações por cima da foto
"""
import argparse
import time
from pathlib import Path

import cv2

from catalog import load_stars, load_constellations
from detection import find_stars
from matching import load_catalog_index, identify
from overlay import draw_result, draw_failure
from segmentation import sky_mask

ROOT = Path(__file__).resolve().parent.parent


def run(path: Path, out_dir: Path, index, stars, consts) -> dict:
    img = cv2.imread(str(path))
    if img is None:
        print(f"{path.name}: não consegui abrir a imagem")
        return {}
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    t0 = time.perf_counter()
    det = find_stars(gray)
    result = identify(det, index, gray.shape)
    dt = time.perf_counter() - t0

    if result:
        names = ", ".join(consts[a]["name"] for a in sorted(result["constellations"])) or "(nenhuma completa)"
        print(f"{path.name}: {names}  [FOV {result['pose'][3]:.0f}°, score {result['score']:.2f}, {dt:.1f} s]")
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
    ap.add_argument("--out", type=Path, default=ROOT / "results" / "demo")
    args = ap.parse_args()

    paths = list(args.images)
    if args.samples:
        paths += sorted((ROOT / "data" / "samples").glob("*.jpg"))
    if not paths:
        ap.error("indica pelo menos uma imagem, ou --samples")

    index, stars, consts = load_catalog_index(), load_stars(), load_constellations()
    ok = sum(bool(run(p, args.out, index, stars, consts)) for p in paths)
    print(f"\n{ok}/{len(paths)} resolvidas. Resultados em {args.out}")


if __name__ == "__main__":
    main()
