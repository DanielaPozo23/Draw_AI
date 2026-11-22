import cv2

# Abrir la cámara predeterminada.
camara = cv2.VideoCapture(0, cv2.CAP_DSHOW)

if not camara.isOpened():
    print("Error: no se pudo abrir la cámara.")
    print("Revisa los permisos de cámara y cierra otras aplicaciones que la estén usando.")
    raise SystemExit

print("Cámara iniciada correctamente.")
print("Presiona la tecla Q para cerrar la ventana.")

while True:
    resultado, imagen = camara.read()

    if not resultado:
        print("Error: no se pudo obtener una imagen de la cámara.")
        break

    # Voltear la imagen para que se vea como un espejo.
    imagen = cv2.flip(imagen, 1)

    cv2.putText(
        imagen,
        "Prueba de camara - Presiona Q para salir",
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2
    )

    cv2.imshow("AirDraw IA - Prueba de camara", imagen)

    # Cerrar al presionar la tecla Q.
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

camara.release()
cv2.destroyAllWindows()