"""
Fase 1: gerar céus sintéticos e verificar visualmente a projeção.

Correr a partir da raiz do projeto:
    python src/catalog.py      # só da primeira vez (descarrega os dados)
    python src/fase1_demo.py
"""
from pathlib import Path
import cv2
import numpy as np

from catalog import load_stars, load_constellations, constellation_center
from synthetic import render_sky, draw_constellations

ROOT = Path(__file__).resolve().parent.parent
OUT_FIG = ROOT / "results" / "figures"
OUT_SYN = ROOT / "data" / "synthetic"
OUT_FIG.mkdir(parents=True, exist_ok=True)
OUT_SYN.mkdir(parents=True, exist_ok=True)

stars = load_stars()
consts = load_constellations()

# 1) Orion "limpo" e com as constelações desenhadas: confirma que a projeção está certa
ra0, dec0 = constellation_center("Ori", stars, consts)
img, truth = render_sky(stars, ra0, dec0, fov_deg=50, roll_deg=0, seed=0)
cv2.imwrite(str(OUT_FIG / "fase1_orion.png"), img)
cv2.imwrite(str(OUT_FIG / "fase1_orion_linhas.png"),
            draw_constellations(img, stars, consts, ra0, dec0, 50, 0))
print(f"Orion: {len(truth)} estrelas visíveis na imagem")

# 2) Mesmo céu, mas difícil: rodado, com ruído, estrelas em falta e falsas estrelas
img2, truth2 = render_sky(stars, ra0, dec0, fov_deg=50, roll_deg=35, mag_limit=4.5,
                          noise_std=8, gradient=90, drop_frac=0.2, n_fake=25, seed=1)
cv2.imwrite(str(OUT_FIG / "fase1_orion_dificil.png"), img2)

# 3) Pequeno dataset aleatório com verdade de referência, para as fases seguintes
rng = np.random.default_rng(42)
for i in range(20):
    ra = rng.uniform(0, 360)
    dec = np.degrees(np.arcsin(rng.uniform(-1, 1)))              # uniforme na esfera
    roll, fov = rng.uniform(0, 360), rng.uniform(35, 70)
    im, tr = render_sky(stars, ra, dec, fov_deg=fov, roll_deg=roll, seed=i)
    name = f"sky_{i:03d}"
    cv2.imwrite(str(OUT_SYN / f"{name}.png"), im)
    tr.to_csv(OUT_SYN / f"{name}_truth.csv", index=False)
    with open(OUT_SYN / f"{name}_params.txt", "w") as f:
        f.write(f"ra0={ra:.4f}\ndec0={dec:.4f}\nroll={roll:.4f}\nfov={fov:.4f}\n")
print(f"20 céus aleatórios guardados em {OUT_SYN}")
