import os
import time

import cv2
import mediapipe as mp


# ---------------------------------------------------------
# CONFIGURACIÓN
# ---------------------------------------------------------

# Obtener la carpeta donde se encuentra este archivo.
CARPETA_PROYECTO = os.path.dirname(os.path.abspath(__file__))

# Ruta del modelo de inteligencia artificial.
RUTA_MODELO = os.path.join(
    CARPETA_PROYECTO,
    "modelos",
    "hand_landmarker.task"
)

# Conexiones entre los 21 puntos de la mano.
CONEXIONES_MANO = [
    # Pulgar
    (0, 1), (1, 2), (2, 3), (3, 4),

    # Índice
    (0, 5), (5, 6), (6, 7), (7, 8),

    # Dedo medio
    (0, 9), (9, 10), (10, 11), (11, 12),

    # Anular
    (0, 13), (13, 14), (14, 15), (15, 16),

    # Meñique
    (0, 17), (17, 18), (18, 19), (19, 20),

    # Palma
    (5, 9), (9, 13), (13, 17), (17, 5)
]


# ---------------------------------------------------------
# VERIFICAR EL MODELO
# ---------------------------------------------------------

if not os.path.isfile(RUTA_MODELO):
    print("ERROR: No se encontró el modelo de detección de manos.")
    print(f"Ruta esperada: {RUTA_MODELO}")
    print("Descarga hand_landmarker.task y colócalo en la carpeta modelos.")
    raise SystemExit


# ---------------------------------------------------------
# CONFIGURAR MEDIAPIPE
# ---------------------------------------------------------

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

opciones = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=RUTA_MODELO),
    running_mode=VisionRunningMode.VIDEO,
    num_hands=1,
    min_hand_detection_confidence=0.5,
    min_hand_presence_confidence=0.5,
    min_tracking_confidence=0.5
)


# ---------------------------------------------------------
# ABRIR CÁMARA
# ---------------------------------------------------------

camara = cv2.VideoCapture(0, cv2.CAP_DSHOW)

if not camara.isOpened():
    print("ERROR: No se pudo abrir la cámara.")
    raise SystemExit

print("Cámara iniciada.")
print("Muestra tu mano frente a la cámara.")
print("Presiona Q para cerrar la ventana.")

tiempo_anterior = -1

try:
    with HandLandmarker.create_from_options(opciones) as detector:

        while True:
            resultado, imagen = camara.read()

            if not resultado:
                print("ERROR: No se pudo obtener una imagen de la cámara.")
                break

            # Voltear la imagen para verla como un espejo.
            imagen = cv2.flip(imagen, 1)

            alto, ancho, _ = imagen.shape

            # MediaPipe necesita la imagen en formato RGB.
            imagen_rgb = cv2.cvtColor(imagen, cv2.COLOR_BGR2RGB)

            imagen_mp = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=imagen_rgb
            )

            # El modo VIDEO requiere marcas de tiempo crecientes.
            tiempo_actual = int(time.monotonic() * 1000)
            tiempo_actual = max(tiempo_actual, tiempo_anterior + 1)
            tiempo_anterior = tiempo_actual

            # Detectar la mano en el fotograma.
            deteccion = detector.detect_for_video(
                imagen_mp,
                tiempo_actual
            )

            if deteccion.hand_landmarks:
                for mano in deteccion.hand_landmarks:

                    puntos = []

                    # Convertir coordenadas normalizadas a píxeles.
                    for punto in mano:
                        x = int(punto.x * ancho)
                        y = int(punto.y * alto)
                        puntos.append((x, y))

                    # Dibujar las líneas entre los puntos.
                    for inicio, fin in CONEXIONES_MANO:
                        cv2.line(
                            imagen,
                            puntos[inicio],
                            puntos[fin],
                            (0, 255, 0),
                            2
                        )

                    # Dibujar los puntos de la mano.
                    for x, y in puntos:
                        cv2.circle(
                            imagen,
                            (x, y),
                            5,
                            (255, 0, 255),
                            -1
                        )

                    # Resaltar la punta del dedo índice: punto 8.
                    x_indice, y_indice = puntos[8]

                    cv2.circle(
                        imagen,
                        (x_indice, y_indice),
                        10,
                        (0, 255, 255),
                        -1
                    )

                    cv2.putText(
                        imagen,
                        "Mano detectada",
                        (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (0, 255, 0),
                        2
                    )

            else:
                cv2.putText(
                    imagen,
                    "Muestra tu mano a la camara",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 0, 255),
                    2
                )

            cv2.putText(
                imagen,
                "Presiona Q para salir",
                (20, alto - 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2
            )

            cv2.imshow("AirDraw IA - Deteccion de mano", imagen)

            # Salir al presionar Q.
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

finally:
    camara.release()
    cv2.destroyAllWindows()

print("Programa finalizado correctamente.")