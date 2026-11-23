"""
Máscara céu / paisagem, para não usar luzes, rochas e ramos como estrelas.

Ideia base: depois de apagar as estrelas, o céu é liso e a paisagem tem textura.
v1 é a predefinida; a v2 (--mask 2) trata bordas pretas e silhuetas escuras
mas nos testes limpos não ficou melhor (ver relatório, secção da máscara v2).
"""
import cv2
import numpy as np


def no_data_mask(gray: np.ndarray, size, level: int = 1, min_frac: float = 0.01,
                 max_stars: float = 2e-4) -> np.ndarray:
    """Bordas pretas de mosaicos: preto exato, ligado à borda e sem estrelas.
    (sem o teste das estrelas, céus com o preto cortado também caíam aqui)
    size = tamanho da imagem reduzida onde se devolve a máscara."""
    # tem de ser na resolução original, ao reduzir o ruído do céu desaparece
    h, w = gray.shape
    flat = cv2.dilate(gray, np.ones((7, 7), np.uint8)) <= level
    flat = cv2.resize(flat.astype(np.uint8), size, interpolation=cv2.INTER_AREA) > 0
    n, lab, stats, _ = cv2.connectedComponentsWithStats(flat.astype(np.uint8))
    edge = set(np.unique(np.concatenate([lab[0], lab[-1], lab[:, 0], lab[:, -1]]))) - {0}
    keep = []
    for i in edge:
        if stats[i, cv2.CC_STAT_AREA] < min_frac * flat.size:
            continue
        inner = cv2.morphologyEx((lab == i).astype(np.uint8), cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))
        inner = cv2.erode(inner, np.ones((7, 7), np.uint8))
        inner = cv2.resize(inner, (w, h), interpolation=cv2.INTER_NEAREST).astype(bool)
        if not inner.any() or (gray[inner] >= 20).mean() <= max_stars:
            keep.append(i)
    return np.isin(lab, keep)


def first_last_valid(valid: np.ndarray):
    """Em cada coluna, a primeira e a última linha com dados (-1 se a coluna não tem)."""
    has = valid.any(0)
    first = np.where(has, valid.argmax(0), -1)
    last = np.where(has, valid.shape[0] - 1 - valid[::-1].argmax(0), -1)
    return first, last


def labels_touching(lab: np.ndarray, rows: np.ndarray) -> list:
    """Etiquetas que aparecem em lab[rows[c], c] (uma linha por coluna)."""
    c = np.where(rows >= 0)[0]
    return list(set(np.unique(lab[rows[c], c])) - {0})


def silhouettes(dark, textured, valid, last, min_edge: float = 0.2) -> np.ndarray:
    """Zonas escuras ligadas ao fundo que acabam numa aresta (árvores, prédios).
    A vinhetagem também é escura mas tem a fronteira lisa, por isso fica de fora."""
    n, lab = cv2.connectedComponents(dark.astype(np.uint8))
    out = np.zeros_like(dark)
    ring_k = np.ones((7, 7), np.uint8)
    for i in labels_touching(lab, last):
        comp = lab == i
        ring = cv2.dilate(comp.astype(np.uint8), ring_k).astype(bool) & ~dark & valid
        if ring.any() and textured[ring].mean() >= min_edge:
            out |= comp
    return out


def sky_mask_v2(gray: np.ndarray, k: float = 3.0, work_width: int = 600, link: int = 1,
             dark_frac: float = 0.4, min_edge: float = 0.2, debug: bool = False):
    """Máscara v2 (True = céu). debug=True devolve também os mapas intermédios."""
    h, w = gray.shape
    s = work_width / w
    small = cv2.resize(gray, (work_width, max(1, round(h * s))), interpolation=cv2.INTER_AREA)

    # bordas sem dados + normalizar contraste
    nodata = no_data_mask(gray, (small.shape[1], small.shape[0]))
    valid = ~nodata
    lo, hi = np.percentile(small[valid], [1, 99.5]) if valid.any() else (0, 255)
    small = np.clip((small.astype(np.float32) - lo) * 255 / max(hi - lo, 1), 0, 255).astype(np.uint8)
    first, last = first_last_valid(valid)

    # apagar as estrelas com uma abertura
    disk = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    bg = cv2.medianBlur(cv2.morphologyEx(small, cv2.MORPH_OPEN, disk), 5).astype(np.float32)

    # textura = desvio padrão local; referência = faixa de cima
    mean = cv2.blur(bg, (15, 15))
    rough = np.sqrt(np.maximum(cv2.blur(bg * bg, (15, 15)) - mean * mean, 0))
    rows = np.arange(bg.shape[0])[:, None]
    band = valid & (rows >= first) & (rows < first + np.maximum((last - first) // 5, 5))
    ref = np.median(rough[band]) if band.any() else np.median(rough)
    smooth = (rough < k * ref + 1.0) & valid
    smooth = cv2.morphologyEx(smooth.astype(np.uint8), cv2.MORPH_OPEN, np.ones((9, 9), np.uint8)).astype(bool)
    textured = ~smooth & valid
    # só os píxeis lisos, senão antenas iluminadas no topo estragavam o nível
    top_sky = band & smooth
    sky_level = np.median(bg[top_sky]) if top_sky.any() else np.median(bg[band] if band.any() else bg)

    # chão = escuro ou texturado, ligado ao fundo
    dark = (bg < dark_frac * sky_level) & valid
    dark = cv2.morphologyEx(dark.astype(np.uint8), cv2.MORPH_OPEN, np.ones((5, 5), np.uint8)).astype(bool)
    dark = silhouettes(dark, textured, valid, last, min_edge)
    land = dark | textured
    if link > 1:
        land = cv2.dilate(land.astype(np.uint8), np.ones((link, link), np.uint8)).astype(bool) & valid
    _, lab = cv2.connectedComponents(land.astype(np.uint8))
    ground = np.isin(lab, labels_touching(lab, last))

    # textura a "flutuar" (Via Láctea, halos) também é céu
    floating = textured & ~ground

    cand = (smooth | floating) & ~ground
    _, lab = cv2.connectedComponents(cand.astype(np.uint8))
    sky = np.isin(lab, labels_touching(lab, first)).astype(np.uint8)

    # afastar do horizonte e das costuras dos mosaicos
    sky = cv2.erode(sky, np.ones((11, 11), np.uint8))
    sky[cv2.dilate(nodata.astype(np.uint8), np.ones((15, 15), np.uint8)).astype(bool)] = 0
    out = cv2.resize(sky, (w, h), interpolation=cv2.INTER_NEAREST).astype(bool)
    if debug:
        return out, {"norm": small, "rough": rough, "nodata": nodata, "dark": dark,
                     "ground": ground, "smooth": smooth, "ref": ref, "sky_level": sky_level}
    return out


def sky_mask_v1(gray: np.ndarray, k: float = 3.0, work_width: int = 600, link: int = 1) -> np.ndarray:
    """Máscara v1, a predefinida (True = céu).
    k: "liso" = textura < k vezes a da faixa de cima. Trabalha a 600 px de largura."""
    h, w = gray.shape
    s = work_width / w
    small = cv2.resize(gray, (work_width, max(1, round(h * s))), interpolation=cv2.INTER_AREA)

    # normalizar, senão numa foto muito escura tudo parece liso
    lo, hi = np.percentile(small, [1, 99.5])
    small = np.clip((small.astype(np.float32) - lo) * 255 / max(hi - lo, 1), 0, 255).astype(np.uint8)

    disk = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    bg = cv2.medianBlur(cv2.morphologyEx(small, cv2.MORPH_OPEN, disk), 5).astype(np.float32)

    # desvio padrão local 15x15
    mean = cv2.blur(bg, (15, 15))
    rough = np.sqrt(np.maximum(cv2.blur(bg * bg, (15, 15)) - mean * mean, 0))
    ref = np.median(rough[:max(5, rough.shape[0] // 5)])        # faixa de cima
    smooth = (rough < k * ref + 1.0).astype(np.uint8)
    smooth = cv2.morphologyEx(smooth, cv2.MORPH_OPEN, np.ones((9, 9), np.uint8))

    # textura que não toca em baixo conta como céu
    textured = 1 - smooth
    linked = cv2.dilate(textured, np.ones((link, link), np.uint8)) if link > 1 else textured
    _, lab = cv2.connectedComponents(linked)
    touches_bottom = np.isin(lab, list(set(np.unique(lab[-1])) - {0}))
    floating = textured.astype(bool) & ~touches_bottom

    # fica o que está ligado ao topo
    _, lab = cv2.connectedComponents((smooth.astype(bool) | floating).astype(np.uint8))
    sky = np.isin(lab, list(set(np.unique(lab[0])) - {0})).astype(np.uint8)

    sky = cv2.erode(sky, np.ones((11, 11), np.uint8))
    return cv2.resize(sky, (w, h), interpolation=cv2.INTER_NEAREST).astype(bool)


DEFAULT_VERSION = 1


def sky_mask(gray: np.ndarray, version: int | None = None, **kw):
    """version 1 (predefinida) ou 2."""
    v = DEFAULT_VERSION if version is None else version
    return (sky_mask_v2 if v == 2 else sky_mask_v1)(gray, **kw)
