import cv2
import numpy as np
from scipy.ndimage import uniform_filter


img = cv2.imread("assents/Imagem_eua.png")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

brilho_medio = gray.mean()
desvio = gray.std()  # contraste geral

def foco_por_regiao(img):
    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Divide em regiões
    regioes = {
        "topo":   gray[:h//2, :],
        "baixo":  gray[h//2:, :],
        "centro": gray[h//4:3*h//4, w//4:3*w//4],
        "borda":  np.concatenate([gray[:, :w//8].flatten(), gray[:, 7*w//8:].flatten()])
    }

    return {k: round(cv2.Laplacian(v, cv2.CV_64F).var(), 2) for k, v in regioes.items()}

def analise_frequencia(gray):
    f = np.fft.fft2(gray)
    fshift = np.fft.fftshift(f)
    magnitude = np.log(np.abs(fshift) + 1)

    # Alta energia em frequências médias pode indicar artefatos de GAN
    h, w = magnitude.shape
    centro = magnitude[h//4:3*h//4, w//4:3*w//4]
    borda  = magnitude.copy()
    borda[h//4:3*h//4, w//4:3*w//4] = 0

    return {
        "energia_centro": round(centro.mean(), 4),
        "energia_borda":  round(borda[borda > 0].mean(), 4),
        "razao":          round(centro.mean() / (borda[borda > 0].mean() + 1e-5), 4),
    }


def ruido_por_regiao(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(float)
    suavizada = uniform_filter(gray, size=3)
    mapa_ruido = np.abs(gray - suavizada)

    h, w = mapa_ruido.shape
    return {
        "ruido_rosto": round(mapa_ruido[h//4:3*h//4, w//4:3*w//4].mean(), 4),
        "ruido_fundo": round(mapa_ruido[:h//8, :].mean(), 4),
        # Suspeito: ruído muito uniforme entre rosto e fundo
        "desvio_ruido": round(mapa_ruido.std(), 4),
    }

print(ruido_por_regiao(img))