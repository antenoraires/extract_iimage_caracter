import os
import cv2
import numpy as np
from scipy.ndimage import uniform_filter



def extrair_caracteristicas_gerais(img: np.ndarray, gray: np.ndarray) -> dict:
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    b, _, r = cv2.split(img)
    s = cv2.split(hsv)[1]
    suavizada = uniform_filter(gray.astype(float), size=3)

    return {
        "brilho":            round(float(gray.mean()), 2),
        "contraste":         round(float(gray.std()), 2),
        "foco_geral":        round(float(cv2.Laplacian(gray, cv2.CV_64F).var()), 2),
        "saturacao":         round(float(s.mean()), 2),
        "temperatura_rb":    round(float(r.mean()) / (float(b.mean()) + 1e-5), 2),
        "ruido_geral":       round(float(np.std(gray.astype(float) - suavizada)), 2),
        "superexpostos_pct": round(float(np.sum(gray > 250)) / gray.size * 100, 2),
        "subexpostos_pct":   round(float(np.sum(gray < 5)) / gray.size * 100, 2),
        "range_dinamico":    int(gray.max() - gray.min()),
    }


def foco_por_regiao(img: np.ndarray) -> dict:
    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    regioes = {
        "foco_topo":   gray[:h // 2, :],
        "foco_baixo":  gray[h // 2:, :],
        "foco_centro": gray[h // 4:3 * h // 4, w // 4:3 * w // 4],
        "foco_borda":  np.concatenate(
            [gray[:, :w // 8].flatten(), gray[:, 7 * w // 8:].flatten()]
        ),
    }

    return {k: round(float(cv2.Laplacian(v, cv2.CV_64F).var()), 2) for k, v in regioes.items()}


def analise_frequencia(gray: np.ndarray) -> dict:
    f = np.fft.fft2(gray)
    fshift = np.fft.fftshift(f)
    magnitude = np.log(np.abs(fshift) + 1)

    h, w = magnitude.shape
    centro = magnitude[h // 4:3 * h // 4, w // 4:3 * w // 4]
    borda = magnitude.copy()
    borda[h // 4:3 * h // 4, w // 4:3 * w // 4] = 0
    borda_vals = borda[borda > 0]

    return {
        "freq_energia_centro": round(float(centro.mean()), 4),
        "freq_energia_borda":  round(float(borda_vals.mean()), 4),
        "freq_razao":          round(float(centro.mean()) / (float(borda_vals.mean()) + 1e-5), 4),
    }


def ruido_por_regiao(img: np.ndarray) -> dict:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(float)
    suavizada = uniform_filter(gray, size=3)
    mapa_ruido = np.abs(gray - suavizada)

    h, w = mapa_ruido.shape
    return {
        "ruido_rosto":  round(float(mapa_ruido[h // 4:3 * h // 4, w // 4:3 * w // 4].mean()), 4),
        "ruido_fundo":  round(float(mapa_ruido[:h // 8, :].mean()), 4),
        "desvio_ruido": round(float(mapa_ruido.std()), 4),
    }


def gradiente_borda(img: np.ndarray) -> dict:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    gradiente = np.sqrt(sobelx ** 2 + sobely ** 2)

    return {
        "gradiente_medio":  round(float(gradiente.mean()), 2),
        "gradiente_max":    round(float(gradiente.max()), 2),
        "picos_anomalos":   int(np.sum(gradiente > gradiente.mean() * 5)),
    }


def consistencia_pele(img: np.ndarray) -> dict:
    ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    cr = ycrcb[:, :, 1]
    cb = ycrcb[:, :, 2]

    mask_pele = (cr > 133) & (cr < 173) & (cb > 77) & (cb < 127)
    pct_pele = mask_pele.sum() / mask_pele.size * 100

    return {
        "pele_pct":    round(float(pct_pele), 2),
        "pele_std_cr": round(float(cr[mask_pele].std()) if mask_pele.any() else 0.0, 2),
        "pele_std_cb": round(float(cb[mask_pele].std()) if mask_pele.any() else 0.0, 2),
    }


def avaliar_alertas(metricas: dict) -> list[str]:
    alertas = []

    diff_foco = abs(metricas.get("foco_centro", 0) - metricas.get("foco_borda", 0))
    if diff_foco > 200:
        alertas.append(f"Foco inconsistente centro/borda: diferença {diff_foco:.0f}")

    if metricas.get("freq_razao", 0) > 3.0:
        alertas.append(f"Artefato GAN no espectro de frequência: razão {metricas['freq_razao']}")

    diff_ruido = abs(metricas.get("ruido_rosto", 0) - metricas.get("ruido_fundo", 0))
    if diff_ruido < 0.3:
        alertas.append(f"Ruído artificialmente uniforme: diferença {diff_ruido:.4f}")

    if metricas.get("picos_anomalos", 0) > 500:
        alertas.append(f"Bordas anômalas detectadas: {metricas['picos_anomalos']} picos")

    if metricas.get("foco_geral", 0) < 50:
        alertas.append(f"Imagem muito borrada (foco geral {metricas['foco_geral']})")

    if metricas.get("superexpostos_pct", 0) > 5:
        alertas.append(f"Superexposição elevada: {metricas['superexpostos_pct']}%")

    return alertas


# ── Pipeline principal ───────────────────────────────────────────────────────

def analisar_imagem(caminho: str) -> dict:
    img = cv2.imread(caminho)
    if img is None:
        raise FileNotFoundError(f"Imagem não encontrada: {caminho}")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    metricas = {}
    metricas.update(extrair_caracteristicas_gerais(img, gray))
    metricas.update(foco_por_regiao(img))
    metricas.update(analise_frequencia(gray))
    metricas.update(ruido_por_regiao(img))
    metricas.update(gradiente_borda(img))
    metricas.update(consistencia_pele(img))

    alertas = avaliar_alertas(metricas)

    return {
        "arquivo":  caminho,
        "suspeito": len(alertas) >= 2,
        "alertas":  alertas,
        "metricas": metricas,
    }


def imprimir_resultado(resultado: dict) -> None:
    print(f"\n{'='*55}")
    print(f"  Arquivo : {resultado['arquivo']}")
    print(f"  Suspeito: {'⚠️  SIM' if resultado['suspeito'] else '✅ NÃO'}")
    print(f"{'='*55}")

    if resultado["alertas"]:
        print("\n  🚨 Alertas:")
        for a in resultado["alertas"]:
            print(f"     • {a}")
    else:
        print("\n  Nenhum alerta encontrado.")

    print("\n  📊 Métricas:")
    for k, v in resultado["metricas"].items():
        print(f"     {k:<25} {v}")
    print()


if __name__ == "__main__":
    for img in os.listdir("assents"):
        path = os.path.join("assents", img)
        resultado = analisar_imagem(path)
        imprimir_resultado(resultado)