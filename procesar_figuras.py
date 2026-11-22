import cv2
import numpy as np


def extraer_caracteristicas(imagen):
    """Convierte un dibujo sobre fondo blanco en medidas de su contorno."""
    if imagen is None:
        raise ValueError("No se pudo leer la imagen.")

    # Separar del fondo los trazos, sin importar su color.
    mascara = (imagen.min(axis=2) < 215).astype(np.uint8) * 255
    puntos = cv2.findNonZero(mascara)

    if puntos is None or cv2.countNonZero(mascara) < 30:
        raise ValueError("Dibuja una figura antes de presionar R.")

    x, y, ancho, alto = cv2.boundingRect(puntos)
    if min(ancho, alto) < 12:
        raise ValueError("Dibuja una figura más grande y completa.")

    # La envolvente rodea el trazo y tolera pequeñas aberturas.
    contorno = cv2.convexHull(puntos)
    area = cv2.contourArea(contorno)
    perimetro = cv2.arcLength(contorno, True)

    if area < 80 or perimetro == 0:
        raise ValueError("El trazo no tiene suficiente área.")

    # El rectángulo puede girar con la figura.
    _, (lado_a, lado_b), _ = cv2.minAreaRect(contorno)
    area_rectangulo = lado_a * lado_b

    if area_rectangulo <= 0:
        raise ValueError("No se pudo medir la figura.")

    caracteristicas = [
        4 * np.pi * area / perimetro**2,
        area / area_rectangulo,
        min(lado_a, lado_b) / max(lado_a, lado_b),
    ]

    # Medir esquinas con varias tolerancias, porque el trazo es manual.
    for tolerancia in (0.015, 0.025, 0.04, 0.06):
        poligono = cv2.approxPolyDP(
            contorno, tolerancia * perimetro, True
        )
        caracteristicas.append(len(poligono))

    return np.array(caracteristicas, dtype=np.float32)