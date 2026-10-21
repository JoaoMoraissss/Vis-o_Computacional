"""
Segmentação céu / paisagem.

Nas fotos reais, rochas, árvores, edifícios e luzes produzem pontos brilhantes
que a deteção confunde com estrelas. Esta máscara diz onde está o céu.

Ideia: depois de apagar as estrelas (abertura morfológica), o céu é uma zona
LISA e a paisagem tem TEXTURA. Mede-se a textura com o desvio padrão local.

Regras (cada uma sozinha falhava num tipo de foto):
  1. Zona lisa ligada ao topo da imagem -> céu.
     (Sozinha, falhava na Via Láctea e nos halos das estrelas brilhantes,
      que também têm textura.)
  2. Zona com textura que NÃO toca na borda de baixo -> também é céu.
     A paisagem está sempre em baixo; algo texturado a "flutuar" no meio
     do céu é Via Láctea, nebulosa ou o halo de uma estrela.
No fim encolhe-se a máscara um pouco para afastar a linha do horizonte.

Antes de tudo normaliza-se o contraste (percentis 1 e 99.5), para uma foto
muito escura se comportar como uma clara.

Limitação conhecida: chão escuro e liso (água, campo de noite) ligado ao céu
por uma zona lisa fica marcado como céu. Como quase não tem deteções, não estraga muito.
"""
import cv2
import numpy as np


def sky_mask(gray: np.ndarray, k: float = 3.0, work_width: int = 600, link: int = 1) -> np.ndarray:
    """Máscara booleana (True = céu), do mesmo tamanho que gray.

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
