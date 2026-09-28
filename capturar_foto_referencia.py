"""
============================================================================
 CAPTURADOR DE FOTOS DE REFERENCIA (desde la webcam)
============================================================================
Herramienta auxiliar: abre la cámara, y cada vez que presionas 'c'
captura el fotograma actual, te pide el nombre de la persona por
consola, y guarda la foto dentro de 'rostros_conocidos/' con el nombre
correcto (agregando automáticamente un sufijo _2, _3... si la persona
ya tiene fotos).

Útil para sumar una foto "en condiciones reales de webcam" a personas
que ya tienen una foto tipo carnet, mejorando mucho la precisión del
reconocimiento en vivo.

Controles:
    c -> capturar foto (te pedirá el nombre en la terminal)
    q -> salir
============================================================================
"""

import os
import sys
import cv2

CARPETA_ROSTROS_CONOCIDOS = "rostros_conocidos"
INDICE_CAMARA = 0


def siguiente_nombre_archivo(nombre_persona: str, carpeta: str) -> str:
    """
    Genera un nombre de archivo único para 'nombre_persona' dentro de
    'carpeta', respetando el formato Nombre_Apellido(_N).jpg
    """
    base = nombre_persona.strip().replace(" ", "_")
    ruta_base = os.path.join(carpeta, f"{base}.jpg")

    if not os.path.exists(ruta_base):
        return ruta_base

    contador = 2
    while True:
        ruta_candidata = os.path.join(carpeta, f"{base}_{contador}.jpg")
        if not os.path.exists(ruta_candidata):
            return ruta_candidata
        contador += 1


def main():
    os.makedirs(CARPETA_ROSTROS_CONOCIDOS, exist_ok=True)

    captura = cv2.VideoCapture(INDICE_CAMARA)
    if not captura.isOpened():
        print(f"[ERROR] No fue posible acceder a la cámara (índice {INDICE_CAMARA}).")
        sys.exit(1)

    print("[INFO] Cámara iniciada.")
    print("       Presiona 'c' para capturar una foto, 'q' para salir.")

    try:
        while True:
            ret, frame = captura.read()
            if not ret:
                print("[ADVERTENCIA] No se pudo leer un fotograma de la cámara.")
                break

            vista = frame.copy()
            cv2.putText(
                vista, "Presiona 'c' para capturar | 'q' para salir",
                (10, 30), cv2.FONT_HERSHEY_DUPLEX, 0.6, (0, 255, 255), 1,
            )
            cv2.imshow("Capturador de fotos de referencia", vista)

            tecla = cv2.waitKey(1) & 0xFF

            if tecla == ord("q"):
                print("[INFO] Cerrando capturador...")
                break

            elif tecla == ord("c"):
                nombre_persona = input(
                    "\n>> Escribe el nombre de la persona (ej: Silvia Aguilar): "
                ).strip()
                if not nombre_persona:
                    print("[ADVERTENCIA] Nombre vacío, foto descartada.")
                    continue

                ruta_destino = siguiente_nombre_archivo(
                    nombre_persona, CARPETA_ROSTROS_CONOCIDOS
                )
                cv2.imwrite(ruta_destino, frame)
                print(f"[INFO] Foto guardada en: {ruta_destino}")

    finally:
        captura.release()
        cv2.destroyAllWindows()
        print("[INFO] Recursos liberados.")


if __name__ == "__main__":
    main()
