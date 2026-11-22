from pathlib import Path

import cv2
import joblib
import numpy as np

from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from procesar_figuras import extraer_caracteristicas


CARPETA = Path(__file__).resolve().parent
DATOS = CARPETA / "datos_figuras"
RUTA_MODELO = CARPETA / "modelos" / "clasificador_figuras.joblib"

CLASES = ["circulo", "cuadrado", "triangulo"]
EXTENSIONES = {".png", ".jpg", ".jpeg"}


def cargar_datos(grupo):
    caracteristicas = []
    etiquetas = []
    cantidades = {}

    for clase in CLASES:
        carpeta = DATOS / grupo / clase
        cantidad = 0

        if not carpeta.is_dir():
            raise FileNotFoundError(
                f"No existe la carpeta: {carpeta}"
            )

        for archivo in sorted(carpeta.iterdir()):
            if archivo.suffix.lower() not in EXTENSIONES:
                continue

            try:
                # Leer también rutas con acentos.
                contenido = np.fromfile(archivo, dtype=np.uint8)
                imagen = cv2.imdecode(
                    contenido, cv2.IMREAD_COLOR
                )

                vector = extraer_caracteristicas(imagen)

                caracteristicas.append(vector)
                etiquetas.append(clase)
                cantidad += 1

            except (ValueError, cv2.error, OSError) as error:
                print(f"Imagen omitida: {archivo.name}: {error}")

        cantidades[clase] = cantidad
        print(f"{grupo} / {clase}: {cantidad} imágenes válidas")

    return (
        np.asarray(caracteristicas, dtype=np.float32),
        np.asarray(etiquetas),
        cantidades
    )


def main():
    print("\nCARGANDO ENTRENAMIENTO")
    x_entrenamiento, y_entrenamiento, cantidades = cargar_datos(
        "entrenamiento"
    )

    if any(cantidad < 5 for cantidad in cantidades.values()):
        raise ValueError(
            "Necesitas al menos 5 imágenes válidas de cada figura "
            "para esta primera prueba."
        )

    print("\nEntrenando el clasificador...")

    modelo = make_pipeline(
        StandardScaler(),
        SVC(kernel="rbf", C=2, gamma="scale", class_weight="balanced")
    )
    modelo.fit(x_entrenamiento, y_entrenamiento)

    RUTA_MODELO.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(modelo, RUTA_MODELO)

    print(f"Modelo guardado en: {RUTA_MODELO}")

    print("\nCARGANDO PRUEBA")
    x_prueba, y_prueba, cantidades_prueba = cargar_datos("prueba")

    if len(y_prueba) == 0:
        print(
            "\nEl modelo está entrenado, pero no se ha evaluado: "
            "las carpetas de prueba están vacías."
        )
        return

    predicciones = modelo.predict(x_prueba)

    aciertos = int(np.sum(predicciones == y_prueba))
    exactitud = accuracy_score(y_prueba, predicciones)

    print(f"\nAciertos: {aciertos} de {len(y_prueba)}")
    print(f"Exactitud en prueba: {exactitud:.1%}")

    for clase in CLASES:
        seleccion = y_prueba == clase
        total = int(np.sum(seleccion))

        if total:
            correctos = int(
                np.sum(predicciones[seleccion] == clase)
            )
            print(f"  {clase}: {correctos}/{total}")
        else:
            print(f"  {clase}: sin ejemplos de prueba")

    print("\nMatriz de confusión")
    print("Orden: circulo, cuadrado, triangulo")
    print("Filas: figura real | Columnas: predicción")
    print(
        confusion_matrix(
            y_prueba,
            predicciones,
            labels=CLASES
        )
    )

    if any(valor == 0 for valor in cantidades_prueba.values()):
        print(
            "\nEvaluación incompleta: faltan ejemplos de prueba "
            "de alguna figura."
        )


if __name__ == "__main__":
    main()