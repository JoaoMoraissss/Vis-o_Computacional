"""
Gerador de imagens sintéticas do céu.

Para um centro (RA, Dec), um campo de visão e uma rotação, projeta o catálogo
na imagem com a projeção gnomónica (a de uma câmara pinhole) e desenha cada
estrela como uma mancha gaussiana com brilho proporcional à magnitude.
Devolve também a verdade de referência: onde está cada estrela na imagem.
"""
import numpy as np
import pandas as pd
import cv2


def gnomonic(ra_deg, dec_deg, ra0_deg, dec0_deg):
    """Projeção gnomónica (plano tangente) em torno de (ra0, dec0).
    Devolve (xi, eta) em unidades de tan(ângulo) e uma máscara das estrelas à frente da câmara."""
    ra, dec = np.radians(ra_deg), np.radians(dec_deg)
    ra0, dec0 = np.radians(ra0_deg), np.radians(dec0_deg)
    cos_c = np.sin(dec0) * np.sin(dec) + np.cos(dec0) * np.cos(dec) * np.cos(ra - ra0)
    front = cos_c > 0
    with np.errstate(divide="ignore", invalid="ignore"):
        xi = np.cos(dec) * np.sin(ra - ra0) / cos_c
        eta = (np.cos(dec0) * np.sin(dec) - np.sin(dec0) * np.cos(dec) * np.cos(ra - ra0)) / cos_c
    return xi, eta, front


def project_to_pixels(stars: pd.DataFrame, ra0, dec0, fov_deg, roll_deg, width, height):
    """Coordenadas em píxeis de todas as estrelas. O campo de visão refere-se à largura da imagem."""
    xi, eta, front = gnomonic(stars["ra_deg"].values, stars["dec_deg"].values, ra0, dec0)
    f = (width / 2) / np.tan(np.radians(fov_deg) / 2)           # distância focal em píxeis
    # No céu, o leste fica à esquerda quando o norte está para cima, por isso xi troca de sinal
    u, v = -f * xi, -f * eta
    r = np.radians(roll_deg)
    x = width / 2 + np.cos(r) * u - np.sin(r) * v
    y = height / 2 + np.sin(r) * u + np.cos(r) * v
    inside = front & (x >= 0) & (x < width) & (y >= 0) & (y < height)
    return x, y, inside


def render_sky(stars: pd.DataFrame, ra0, dec0, fov_deg=60, roll_deg=0,
               width=1280, height=960, mag_limit=5.0,
               noise_std=4.0, gradient=40.0, drop_frac=0.0, n_fake=0, seed=None):
    """
    Gera uma imagem 8 bits do céu e a tabela de verdade de referência.

    mag_limit  : magnitude mais fraca visível (telemóvel na cidade ~4, céu escuro ~5.5)
    noise_std  : ruído gaussiano do sensor
    gradient   : intensidade do gradiente de poluição luminosa
    drop_frac  : fração de estrelas visíveis removidas ao acaso (nuvens, oclusões)
    n_fake     : número de falsas estrelas (pixéis quentes, aviões, luzes)
    """
    rng = np.random.default_rng(seed)
    vis = stars[stars["mag"] <= mag_limit].copy()
    x, y, inside = project_to_pixels(vis, ra0, dec0, fov_deg, roll_deg, width, height)
    vis["x"], vis["y"] = x, y
    vis = vis[inside]
    if drop_frac > 0:
        vis = vis[rng.random(len(vis)) >= drop_frac]

    img = np.zeros((height, width), np.float32)

    # Fluxo relativo: cada magnitude a menos é ~2.512x mais brilho (escala de Pogson)
    flux = 10 ** (-0.4 * (vis["mag"].values - mag_limit))
    for (xs, ys, fl) in zip(vis["x"].values, vis["y"].values, flux):
        sigma = 1.0 + 0.35 * np.log1p(fl)                         # estrelas brilhantes parecem maiores
        amp = min(255.0, 25.0 * fl)
        r = int(4 * sigma) + 1
        x0, y0 = int(round(xs)), int(round(ys))
        xs_grid, ys_grid = np.meshgrid(np.arange(x0 - r, x0 + r + 1), np.arange(y0 - r, y0 + r + 1))
        g = amp * np.exp(-((xs_grid - xs) ** 2 + (ys_grid - ys) ** 2) / (2 * sigma ** 2))
        ok = (xs_grid >= 0) & (xs_grid < width) & (ys_grid >= 0) & (ys_grid < height)
        np.add.at(img, (ys_grid[ok], xs_grid[ok]), g[ok])

    for _ in range(n_fake):                                        # falsas estrelas
        fx, fy = rng.integers(0, width), rng.integers(0, height)
        img[fy, fx] += rng.uniform(60, 200)

    # Poluição luminosa: gradiente mais claro junto ao "horizonte" (fundo da imagem)
    yy = np.linspace(0, 1, height)[:, None]
    img += gradient * yy ** 2 + 10
    img += rng.normal(0, noise_std, img.shape)

    img = np.clip(img, 0, 255).astype(np.uint8)
    truth = vis[["hip", "proper", "con", "mag", "x", "y"]].reset_index(drop=True)
    return img, truth


def draw_constellations(img, stars, consts, ra0, dec0, fov_deg, roll_deg, with_names=True):
    """Desenha as linhas das constelações por cima de uma imagem (para verificar a projeção)."""
    h, w = img.shape[:2]
    out = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR) if img.ndim == 2 else img.copy()
    x, y, inside = project_to_pixels(stars, ra0, dec0, fov_deg, roll_deg, w, h)
    pos = {hip: (x[i], y[i], inside[i]) for i, hip in enumerate(stars["hip"].values)}
    front = gnomonic(stars["ra_deg"].values, stars["dec_deg"].values, ra0, dec0)[2]
    front = dict(zip(stars["hip"].values, front))

    for abbr, c in consts.items():
        pts_drawn = []
        for line in c["lines"]:
            for a, b in zip(line[:-1], line[1:]):
                if a not in pos or b not in pos or not (front[a] and front[b]):
                    continue
                pa, pb = pos[a], pos[b]
                if not (pa[2] or pb[2]):
                    continue
                p1 = (int(pa[0]), int(pa[1])); p2 = (int(pb[0]), int(pb[1]))
                cv2.line(out, p1, p2, (255, 180, 80), 1, cv2.LINE_AA)
                pts_drawn += [pa[:2], pb[:2]]
        if with_names and pts_drawn:
            cx, cy = np.mean(pts_drawn, axis=0)
            if 0 <= cx < w and 0 <= cy < h:
                cv2.putText(out, c["name"], (int(cx), int(cy)), cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, (120, 220, 255), 1, cv2.LINE_AA)
    return out
