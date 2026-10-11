"""
Pré-processamento: tirar o fundo do céu e reduzir o ruído.

A imagem é vista como  estrelas + fundo + ruído. O fundo (poluição luminosa)
varia devagar, por isso estima-se com uma mediana de janela grande e subtrai-se.
O ruído varia de píxel para píxel, por isso atenua-se com um filtro gaussiano.
"""

import cv2
import numpy as np 

def estimate_background(gray: np.ndarray, ksize, int=51) -> np.ndarray:
    return cv2.medianBlur(gray, ksize)


def preprocess(gray: np.ndarray, bg_ksize: int = 51, sigma: float = 1.0) -> np.ndarray: #float = 1.0 pode vir a ser mudado
    background = estimate_background(gray, bg_ksize)
    flat = gray.astype(np.float32) - background.astype(np.float32)
    return cv2.GaussianBlur(flat, (0, 0), sigma)

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