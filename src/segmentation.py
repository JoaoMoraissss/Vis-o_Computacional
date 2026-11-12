"""
Segmentação céu / paisagem.

Há duas versões. A 1 é a predefinida; a 2 escolhe-se com sky_mask(..., version=2)
ou mudando DEFAULT_VERSION (python src/evaluate.py --mask 2 faz isso).

Porque não é a 2 a predefinida: corrige casos que a 1 falha (mosaicos com bordas
pretas, silhuetas escuras de árvores e prédios, céu com o preto cortado), mas nos
conjuntos de teste limpos não ficou melhor: test2 6/8 contra 7/8 (antes de uma
correção), test3 2/7 contra 3/7. Com tão poucas fotos a diferença é ruído, mas
também não há prova de melhoria. Números em results/metrics/ e results/metrics/mascara_v2/.

VERSÃO 1: regras 1 e 2 abaixo, sobre a textura.
VERSÃO 2: regras 1, 2 e 3, mais o tratamento das zonas sem dados.

Nas fotos reais, rochas, árvores, edifícios e luzes produzem pontos brilhantes
que a deteção confunde com estrelas. Esta máscara diz onde está o céu.

Ideia: depois de apagar as estrelas (abertura morfológica), o céu é uma zona
LISA e a paisagem tem TEXTURA. Mede-se a textura com o desvio padrão local.

Regras (cada uma sozinha falhava num tipo de foto):
  1. Zona lisa ligada ao topo -> céu.
     (Sozinha, falhava na Via Láctea e nos halos das estrelas brilhantes,
      que também têm textura.)
  2. Zona com textura que NÃO está ligada ao chão -> também é céu.
     A paisagem está sempre em baixo; algo texturado a "flutuar" no meio
     do céu é Via Láctea, nebulosa ou o halo de uma estrela.
  3. Chão = o que é ESCURO ou TEXTURADO e está ligado à borda de baixo.
     À noite o chão é uma silhueta mais escura que o céu (o céu tem brilho
     do ar e poluição luminosa). Sem esta regra, uma silhueta preta e lisa
     (árvores, prédios) passava por céu liso, e a faixa texturada do
     horizonte, por não tocar na borda de baixo, "flutuava" e virava céu.
     Num céu muito escuro (astrofotos, ou chão iluminado sob céu preto) o
     limiar de "escuro" fica perto de 0 e esta regra praticamente desliga-se.
     Para isso o nível do céu mede-se só nos píxeis lisos da faixa de cima.
     Só a regra do escuro não chega: a vinhetagem também escurece os cantos.
     Por isso uma zona escura só é chão se acabar numa aresta (silhouettes).
No fim encolhe-se a máscara um pouco para afastar a linha do horizonte.

Zonas sem dados: mosaicos e fotos endireitadas têm bordas pretas (valor 0).
Ficam fora de tudo: da normalização, da referência de textura e da máscara.
O "topo" e o "fundo" passam a ser, em cada coluna, o primeiro e o último
píxel COM dados, e não as linhas da imagem.

Antes de tudo normaliza-se o contraste (percentis 1 e 99.5, só nos píxeis com
dados), para uma foto muito escura se comportar como uma clara.

Limitação conhecida: o chão iluminado e liso (neve, areia ao luar) não é
escuro; se a linha do horizonte tiver falhas, pode continuar a passar por céu.
"""
import cv2
import numpy as np


def no_data_mask(gray: np.ndarray, size, level: int = 1, min_frac: float = 0.01,
                 max_stars: float = 2e-4) -> np.ndarray:
    """Bordas sem dados: zonas grandes de PRETO PLANO, ligadas à borda da imagem
    e SEM ESTRELAS.

    Preto plano mede-se na resolução original: o píxel e todos os vizinhos numa
    janela 7x7 estão a <= level. Um céu escuro normal tem ruído e escapa a isto;
    a borda de um mosaico é 0 exato. (Na imagem reduzida não dava: a média da
    redução apaga o ruído.)

    Mas há fotos processadas com o preto cortado: o céu entre as estrelas fica
    a 0 exato e parecia uma borda. A diferença é que esse céu tem estrelas.
    Fechando os buracos que as estrelas deixam na zona, conta-se a fração de
    píxeis brilhantes (>= 20) lá dentro. Medido: bordas de mosaico <= 0.2 por
    10 mil; céu cortado ~40 por 10 mil. Acima de max_stars não é "sem dados".
    size: (largura, altura) da imagem reduzida onde se devolve a máscara.
    """
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
    """Das zonas escuras ligadas ao fundo, fica só as que acabam numa ARESTA.

    Uma silhueta (árvore, prédio, monte) corta o céu de forma brusca, e essa
    fronteira aparece como textura. A vinhetagem da lente também escurece os
    cantos, mas aos poucos: a fronteira é lisa. Medido nas fotos: silhuetas
    com 25% a 100% da fronteira texturada; vinhetagem com ~4%.
    """
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
    """VERSÃO 2 (opcional). Máscara booleana (True = céu), do mesmo tamanho que gray.

    k: uma zona é "lisa" se a textura for menor que k vezes a textura típica
       da faixa de cima da imagem (que assumimos ser quase toda céu).
    work_width: trabalha numa versão reduzida (mais rápido e a textura das
       estrelas individuais desaparece).
    dark_frac: "escuro" = fundo abaixo de dark_frac x o nível típico do céu
       (mediana da faixa de cima).
    min_edge: uma zona escura só é chão se pelo menos esta fração da sua fronteira
       for texturada (ver silhouettes).
    debug: devolve também um dicionário com os mapas intermédios.
    """
    h, w = gray.shape
    s = work_width / w
    small = cv2.resize(gray, (work_width, max(1, round(h * s))), interpolation=cv2.INTER_AREA)

    # 0. zonas sem dados e normalização do contraste (só com os píxeis com dados)
    nodata = no_data_mask(gray, (small.shape[1], small.shape[0]))
    valid = ~nodata
    lo, hi = np.percentile(small[valid], [1, 99.5]) if valid.any() else (0, 255)
    small = np.clip((small.astype(np.float32) - lo) * 255 / max(hi - lo, 1), 0, 255).astype(np.uint8)
    first, last = first_last_valid(valid)

    # 1. apagar as estrelas: a abertura remove tudo o que é menor que o disco
    disk = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    bg = cv2.medianBlur(cv2.morphologyEx(small, cv2.MORPH_OPEN, disk), 5).astype(np.float32)

    # 2. textura = desvio padrão local, sqrt(E[x²] - E[x]²) numa janela 15x15.
    #    Referência: a faixa de cima COM dados (1/5 da altura útil de cada coluna).
    mean = cv2.blur(bg, (15, 15))
    rough = np.sqrt(np.maximum(cv2.blur(bg * bg, (15, 15)) - mean * mean, 0))
    rows = np.arange(bg.shape[0])[:, None]
    band = valid & (rows >= first) & (rows < first + np.maximum((last - first) // 5, 5))
    ref = np.median(rough[band]) if band.any() else np.median(rough)
    smooth = (rough < k * ref + 1.0) & valid
    smooth = cv2.morphologyEx(smooth.astype(np.uint8), cv2.MORPH_OPEN, np.ones((9, 9), np.uint8)).astype(bool)
    textured = ~smooth & valid
    # nível do céu: só os píxeis LISOS da faixa de cima. Com a faixa toda, um
    # objeto iluminado no topo (antenas, um edifício) subia o nível, e o céu
    # preto verdadeiro passava a contar como "escuro" (= chão).
    top_sky = band & smooth
    sky_level = np.median(bg[top_sky]) if top_sky.any() else np.median(bg[band] if band.any() else bg)

    # 3. regra 3: chão = (escuro OU texturado) ligado ao fundo útil de cada coluna
    dark = (bg < dark_frac * sky_level) & valid
    dark = cv2.morphologyEx(dark.astype(np.uint8), cv2.MORPH_OPEN, np.ones((5, 5), np.uint8)).astype(bool)
    dark = silhouettes(dark, textured, valid, last, min_edge)
    land = dark | textured
    if link > 1:
        land = cv2.dilate(land.astype(np.uint8), np.ones((link, link), np.uint8)).astype(bool) & valid
    _, lab = cv2.connectedComponents(land.astype(np.uint8))
    ground = np.isin(lab, labels_touching(lab, last))

    # 4. regra 2: texturado que não é chão "flutua" -> conta como céu
    floating = textured & ~ground

    # 5. regra 1: fica a região (lisa ou flutuante, e não chão) ligada ao topo útil
    cand = (smooth | floating) & ~ground
    _, lab = cv2.connectedComponents(cand.astype(np.uint8))
    sky = np.isin(lab, labels_touching(lab, first)).astype(np.uint8)

    # 6. afastar do horizonte e das bordas sem dados (a junção das fotos de um
    #    mosaico é uma costura com textura falsa), e voltar ao tamanho original
    sky = cv2.erode(sky, np.ones((11, 11), np.uint8))
    sky[cv2.dilate(nodata.astype(np.uint8), np.ones((15, 15), np.uint8)).astype(bool)] = 0
    out = cv2.resize(sky, (w, h), interpolation=cv2.INTER_NEAREST).astype(bool)
    if debug:
        return out, {"norm": small, "rough": rough, "nodata": nodata, "dark": dark,
                     "ground": ground, "smooth": smooth, "ref": ref, "sky_level": sky_level}
    return out


def sky_mask_v1(gray: np.ndarray, k: float = 3.0, work_width: int = 600, link: int = 1) -> np.ndarray:
    """VERSÃO 1 (predefinida). Máscara booleana (True = céu), do mesmo tamanho que gray.

    k: uma zona é "lisa" se a textura for menor que k vezes a textura típica
       da faixa de cima da imagem (que assumimos ser quase toda céu).
    work_width: trabalha numa versão reduzida (mais rápido e a textura das
       estrelas individuais desaparece).
    """
    h, w = gray.shape
    s = work_width / w
    small = cv2.resize(gray, (work_width, max(1, round(h * s))), interpolation=cv2.INTER_AREA)

    # 0. normalizar o contraste: numa foto muito escura toda a textura fica abaixo
    #    de 1 nível de cinzento e a paisagem passava por céu liso
    lo, hi = np.percentile(small, [1, 99.5])
    small = np.clip((small.astype(np.float32) - lo) * 255 / max(hi - lo, 1), 0, 255).astype(np.uint8)

    # 1. apagar as estrelas: a abertura remove tudo o que é menor que o disco
    disk = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    bg = cv2.medianBlur(cv2.morphologyEx(small, cv2.MORPH_OPEN, disk), 5).astype(np.float32)

    # 2. textura = desvio padrão local, sqrt(E[x²] - E[x]²) numa janela 15x15
    mean = cv2.blur(bg, (15, 15))
    rough = np.sqrt(np.maximum(cv2.blur(bg * bg, (15, 15)) - mean * mean, 0))
    ref = np.median(rough[:max(5, rough.shape[0] // 5)])        # faixa de cima
    smooth = (rough < k * ref + 1.0).astype(np.uint8)
    smooth = cv2.morphologyEx(smooth, cv2.MORPH_OPEN, np.ones((9, 9), np.uint8))

    # 3. regra 2: zonas texturadas que não tocam na borda de baixo contam como céu
    #    (antes, liga zonas texturadas próximas com uma dilatação: a copa de uma
    #     árvore tem buracos lisos entre os ramos e ficava "separada" do tronco)
    textured = 1 - smooth
    linked = cv2.dilate(textured, np.ones((link, link), np.uint8)) if link > 1 else textured
    _, lab = cv2.connectedComponents(linked)
    touches_bottom = np.isin(lab, list(set(np.unique(lab[-1])) - {0}))
    floating = textured.astype(bool) & ~touches_bottom

    # 4. regra 1: fica a região (lisa ou "flutuante") ligada ao topo
    _, lab = cv2.connectedComponents((smooth.astype(bool) | floating).astype(np.uint8))
    sky = np.isin(lab, list(set(np.unique(lab[0])) - {0})).astype(np.uint8)

    # 5. afastar do horizonte e voltar ao tamanho original
    sky = cv2.erode(sky, np.ones((11, 11), np.uint8))
    return cv2.resize(sky, (w, h), interpolation=cv2.INTER_NEAREST).astype(bool)


DEFAULT_VERSION = 1


def sky_mask(gray: np.ndarray, version: int | None = None, **kw):
    """Máscara do céu (True = céu). version: 1 (predefinida) ou 2; ver o início do ficheiro."""
    v = DEFAULT_VERSION if version is None else version
    return (sky_mask_v2 if v == 2 else sky_mask_v1)(gray, **kw)
