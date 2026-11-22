import os
import time

import cv2
import mediapipe as mp
import numpy as np

import joblib
from procesar_figuras import extraer_caracteristicas

# ---------------------------------------------------------
# CONFIGURACIÓN
# ---------------------------------------------------------

CARPETA = os.path.dirname(os.path.abspath(__file__))
RUTA_MODELO = os.path.join(
    CARPETA, "modelos", "hand_landmarker.task"
)

ALTO_BARRA = 95
GROSOR = 6
RADIO_BORRADOR = 25
modo_borrador = False

CARPETA_DIBUJOS = os.path.join(CARPETA, "dibujos")
os.makedirs(CARPETA_DIBUJOS, exist_ok=True)

# Mantener el cursor sobre un color este tiempo para elegirlo.
TIEMPO_SELECCION = 0.45

# Valor pequeño = más suavizado; valor grande = más rapidez.
SUAVIZADO = 0.40

COLORES = [
    ("Verde", (0, 255, 0)),
    ("Azul", (255, 100, 0)),
    ("Rojo", (0, 0, 255)),
    ("Amarillo", (0, 255, 255)),
    ("Morado", (255, 0, 255)),
]

color_elegido = 0
RUTA_CLASIFICADOR = os.path.join(
    CARPETA, "modelos", "clasificador_figuras.joblib"
)

clasificador = joblib.load(RUTA_CLASIFICADOR)
prediccion_figura = "Sin analizar"
punto_anterior = None
punto_suave = None

color_candidato = None
inicio_seleccion = 0.0


# ---------------------------------------------------------
# FUNCIONES
# ---------------------------------------------------------

def dedo_extendido(mano, punta, articulacion):
    """Regla sencilla para una mano aproximadamente vertical."""
    return mano[punta].y < mano[articulacion].y - 0.025


def crear_botones(ancho):
    margen = 10
    separacion = 8
    ancho_boton = (
        ancho - 2 * margen - separacion * (len(COLORES) - 1)
    ) // len(COLORES)

    botones = []

    for i in range(len(COLORES)):
        x1 = margen + i * (ancho_boton + separacion)
        x2 = x1 + ancho_boton
        botones.append((x1, 12, x2, 65))

    return botones


def boton_en_posicion(x, y, botones):
    for i, (x1, y1, x2, y2) in enumerate(botones):
        if x1 <= x <= x2 and y1 <= y <= y2:
            return i

    return None


# ---------------------------------------------------------
# MODELO Y CÁMARA
# ---------------------------------------------------------

if not os.path.isfile(RUTA_MODELO):
    print("No se encontró el modelo:")
    print(RUTA_MODELO)
    raise SystemExit

opciones = mp.tasks.vision.HandLandmarkerOptions(
    base_options=mp.tasks.BaseOptions(
        model_asset_path=RUTA_MODELO
    ),
    running_mode=mp.tasks.vision.RunningMode.VIDEO,
    num_hands=1,
    min_hand_detection_confidence=0.5,
    min_hand_presence_confidence=0.5,
    min_tracking_confidence=0.5,
)

camara = cv2.VideoCapture(0, cv2.CAP_DSHOW)

if not camara.isOpened():
    print("No se pudo abrir la cámara.")
    raise SystemExit

lienzo = None
ultimo_tiempo = -1

print("AirDraw IA")
print("Solo indice: dibujar")
print("Indice y medio: mover y seleccionar colores")
print("B: borrar | Q: salir")


# ---------------------------------------------------------
# PROGRAMA PRINCIPAL
# ---------------------------------------------------------

try:
    with mp.tasks.vision.HandLandmarker.create_from_options(
        opciones
    ) as detector:

        while True:
            exito, imagen = camara.read()

            if not exito:
                print("No se pudo obtener imagen de la cámara.")
                break

            imagen = cv2.flip(imagen, 1)
            alto, ancho = imagen.shape[:2]

            if lienzo is None or lienzo.shape != imagen.shape:
                lienzo = np.zeros_like(imagen)
                punto_anterior = None
                punto_suave = None

            botones = crear_botones(ancho)

            rgb = cv2.cvtColor(imagen, cv2.COLOR_BGR2RGB)
            imagen_mp = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=rgb
            )

            tiempo_ms = max(
                int(time.monotonic() * 1000),
                ultimo_tiempo + 1
            )
            ultimo_tiempo = tiempo_ms

            resultado = detector.detect_for_video(
                imagen_mp, tiempo_ms
            )

            estado = "Muestra tu mano"
            cursor = None
            sobre_boton = None
            progreso = 0.0

            if resultado.hand_landmarks:
                mano = resultado.hand_landmarks[0]

                x = int(np.clip(mano[8].x * ancho, 0, ancho - 1))
                y = int(np.clip(mano[8].y * alto, 0, alto - 1))
                cursor = (x, y)

                indice = dedo_extendido(mano, 8, 6)
                medio = dedo_extendido(mano, 12, 10)
                anular = dedo_extendido(mano, 16, 14)
                menique = dedo_extendido(mano, 20, 18)

                # El pulgar puede estar en cualquier posición.
                modo_dibujo = (
                    indice and not medio
                    and not anular and not menique
                )
                modo_seleccion = (
                    indice and medio
                    and not anular and not menique
                )

                if modo_seleccion:
                    estado = "Seleccion: mueve el cursor"
                    punto_anterior = None
                    punto_suave = None

                    sobre_boton = boton_en_posicion(
                        x, y, botones
                    )

                    if sobre_boton is not None:
                        ahora = time.monotonic()

                        if color_candidato != sobre_boton:
                            color_candidato = sobre_boton
                            inicio_seleccion = ahora

                        progreso = min(
                            (ahora - inicio_seleccion)
                            / TIEMPO_SELECCION,
                            1.0
                        )

                        estado = "Manten el cursor para elegir"

                        if progreso >= 1.0:
                            color_elegido = sobre_boton
                            estado = (
                                "Color: "
                                + COLORES[color_elegido][0]
                            )
                    else:
                        color_candidato = None
                elif modo_dibujo and y > ALTO_BARRA:
                    color_candidato = None

                    actual = np.array([x, y], dtype=float)

                    if punto_suave is None:
                        punto_suave = actual
                    else:
                        punto_suave = (
                            SUAVIZADO * actual
                            + (1 - SUAVIZADO) * punto_suave
                        )

                    punto = tuple(
                        punto_suave.astype(int).tolist()
                    )

                    if modo_borrador:
                        estado = "Borrando"

                        # Unir posiciones para borrar de forma continua.
                        if punto_anterior is not None:
                            cv2.line(
                                lienzo,
                                punto_anterior,
                                punto,
                                (0, 0, 0),
                                RADIO_BORRADOR * 2
                            )

                        cv2.circle(
                            lienzo,
                            punto,
                            RADIO_BORRADOR,
                            (0, 0, 0),
                            -1
                        )

                    else:
                        estado = "Dibujando"
                        color = COLORES[color_elegido][1]

                        if punto_anterior is not None:
                            cv2.line(
                                lienzo,
                                punto_anterior,
                                punto,
                                color,
                                GROSOR,
                                cv2.LINE_AA
                            )

                        cv2.circle(
                            lienzo,
                            punto,
                            GROSOR // 2,
                            color,
                            -1
                        )

                    punto_anterior = punto
                    cursor = punto

                else:
                    estado = "Pausa"
                    punto_anterior = None
                    punto_suave = None
                    color_candidato = None

            else:
                punto_anterior = None
                punto_suave = None
                color_candidato = None

            # Superponer los trazos sobre la cámara.
            mascara = np.any(lienzo != 0, axis=2)
            imagen[mascara] = lienzo[mascara]

            # Barra de herramientas.
            cv2.rectangle(
                imagen,
                (0, 0),
                (ancho - 1, ALTO_BARRA),
                (30, 30, 30),
                -1
            )

            for i, (x1, y1, x2, y2) in enumerate(botones):
                nombre, color = COLORES[i]

                cv2.rectangle(
                    imagen, (x1, y1), (x2, y2), color, -1
                )

                if i == color_elegido:
                    cv2.rectangle(
                        imagen,
                        (x1 - 2, y1 - 2),
                        (x2 + 2, y2 + 2),
                        (255, 255, 255),
                        3
                    )

                cv2.putText(
                    imagen,
                    nombre,
                    (x1 + 5, y1 + 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    (0, 0, 0),
                    1,
                    cv2.LINE_AA
                )

                if i == sobre_boton:
                    fin = x1 + int((x2 - x1) * progreso)
                    cv2.rectangle(
                        imagen,
                        (x1, y2 - 6),
                        (fin, y2),
                        (255, 255, 255),
                        -1
                    )

            cv2.putText(
                imagen,
                f"2 dedos: color | E: lapiz/borrador | S: guardar | B: limpiar | Q: salir | R: reconocer figura",
                (10, 85),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.35,
                (255, 255, 255),
                1,
                cv2.LINE_AA
            )

                        # El cursor cambia de tamaño según la herramienta.
            if cursor is not None:
                radio = RADIO_BORRADOR if modo_borrador else 11

                cv2.circle(
                    imagen, cursor, radio, (0, 0, 0), 4
                )
                cv2.circle(
                    imagen, cursor, radio, (255, 255, 255), 2
                )

            cv2.rectangle(
                imagen,
                (0, alto - 32),
                (ancho - 1, alto - 1),
                (30, 30, 30),
                -1
            )

            cv2.putText(
                imagen,
                                f"{'BORRADOR' if modo_borrador else 'LAPIZ'} | {estado}",
                (10, alto - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                1,
                cv2.LINE_AA
            )
            
            cv2.rectangle(
                imagen,
                (5, ALTO_BARRA + 5),
                (ancho - 5, ALTO_BARRA + 40),
                (30, 30, 30),
                -1
            )

            cv2.putText(
                imagen,
                f"IA: {prediccion_figura} | R: reconocer",
                (12, ALTO_BARRA + 29),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                1,
                cv2.LINE_AA
            )


            cv2.imshow("AirDraw IA", imagen)

            tecla = cv2.waitKey(1) & 0xFF
            
            if tecla in (ord("r"), ord("R")):
                try:
                    # Preparar el lienzo como las imágenes guardadas.
                    dibujo = np.full_like(lienzo, 255)
                    mascara_dibujo = np.any(lienzo != 0, axis=2)
                    dibujo[mascara_dibujo] = lienzo[mascara_dibujo]

                    vector = extraer_caracteristicas(dibujo)
                    resultado_figura = clasificador.predict(
                        [vector]
                    )[0]

                    prediccion_figura = resultado_figura
                    print(f"Figura reconocida: {resultado_figura}")

                except ValueError as error:
                    prediccion_figura = "Dibuja una figura primero"
                    print(error)

            if tecla in (ord("q"), ord("Q")):
                break

            elif tecla in (ord("b"), ord("B")):
                lienzo[:] = 0
                punto_anterior = None
                punto_suave = None
                print("Lienzo borrado.")

            elif tecla in (ord("e"), ord("E")):
                modo_borrador = not modo_borrador

                # Evitar conectar trazos al cambiar de herramienta.
                punto_anterior = None
                punto_suave = None

                herramienta = (
                    "Borrador" if modo_borrador else "Lapiz"
                )
                print(f"Herramienta: {herramienta}")

            elif tecla in (ord("s"), ord("S")):
                # Guardar únicamente el dibujo sobre fondo blanco.
                # No se incluye la imagen de la cámara.
                dibujo = np.full_like(lienzo, 255)
                pixeles_dibujados = np.any(lienzo != 0, axis=2)
                dibujo[pixeles_dibujados] = lienzo[pixeles_dibujados]

                nombre = (
                    "AirDraw_"
                    + time.strftime("%Y%m%d_%H%M%S")
                    + f"_{time.time_ns() % 1_000_000_000:09d}.png"
                )

                ruta = os.path.join(CARPETA_DIBUJOS, nombre)

                try:
                    # Permite guardar también en rutas con acentos.
                    correcto, datos = cv2.imencode(".png", dibujo)

                    if not correcto:
                        raise RuntimeError(
                            "No se pudo convertir el dibujo a PNG."
                        )

                    datos.tofile(ruta)
                    print(f"Dibujo guardado en: {ruta}")

                except Exception as error:
                    print(f"No se pudo guardar: {error}")

finally:
    camara.release()
    cv2.destroyAllWindows()

print("AirDraw IA finalizado correctamente.")