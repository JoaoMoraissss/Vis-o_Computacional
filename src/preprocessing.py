"""
Pré-processamento: tirar o fundo do céu e reduzir o ruído.

A imagem é vista como  estrelas + fundo + ruído. O fundo (poluição luminosa)
varia devagar, por isso estima-se com uma mediana de janela grande e subtrai-se.
O ruído varia de píxel para píxel, por isso atenua-se com um filtro gaussiano.
"""

import cv2
import numpy as np 

def estimate_background(gray: np.ndarray, ksize: int = 51) -> np.ndarray:
    return cv2.medianBlur(gray, ksize)


def preprocess(gray: np.ndarray, bg_ksize: int = 51, sigma: float = 1.0) -> np.ndarray: #float = 1.0 pode vir a ser mudado
    background = estimate_background(gray, bg_ksize)
    flat = gray.astype(np.float32) - background.astype(np.float32)
    return cv2.GaussianBlur(flat, (0, 0), sigma)


def tophat(gray: np.ndarray, r_frac: float = 0.004, sigma: float = 1.0):
    """Alternativa a preprocess, melhor para fotos reais: top-hat morfológico.

    top-hat = imagem - abertura(imagem, disco de raio r).
    A abertura apaga tudo o que é menor que o disco (as estrelas) e mantém o que
    é maior (paisagem, nuvens, Via Láctea, gradientes de luz). Subtraindo, sobra
    só o que é pequeno: as estrelas. A mediana de 51 px de preprocess deixava
    passar estruturas médias, como rochas e ramos.

    r_frac: raio do disco em fração da largura (0.004 -> ~5 px em 1280, ~10 px em 2400),
            para funcionar com qualquer resolução.
    No fim subtrai-se a mediana: num céu com ruído, o top-hat é sempre >= 0
    (é "píxel - mínimo local"), e sem isto o fundo inteiro ficava acima do limiar.

    Devolve (imagem filtrada, r em píxeis).
    """
    r = max(3, round(r_frac * gray.shape[1]))
    disk = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
    th = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, disk).astype(np.float32)
    th = cv2.GaussianBlur(th, (0, 0), sigma)
    return th - np.median(th), r

def noise_level(img:np.ndarray)-> float:
     return float(1.4826 * np.median(np.abs(img - np.median(img))))

"""
se mexerem aqui tenham em atenção: 
Converter para float32 antes de subtrair. 
    Em uint8 os valores negativos dariam a volta (3 − 5 = 254). 
    Em float o ruído fica centrado em zero, com valores positivos e negativos.
sigma = 1.0 é aproximadamente o tamanho de uma estrela fraca na 
    imagemou video, mais tarde ainda vamos ver se vamos meter em video, pode ser que venhamos a ter que mudar. 
Filtrar com a forma do que procuras é o que maximiza o contraste entre estrela e ruído.
noise_level vai servir na deteção, para pôr o limiar em "k vezes o ruído" em vez de um número fixo. 
    Usa a mediana dos desvios (MAD) em vez do desvio padrão normal porque as estrelas inflacionariam este último. 
    O 1.4826 converte MAD em desvio padrão para ruído gaussiano.
"""