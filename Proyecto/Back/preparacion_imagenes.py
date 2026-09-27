import cv2
import numpy as np

def preparacion_img(ruta_imagen):
    imagen = cv2.imread(ruta_imagen)
    if imagen is None:
        print(f"Error: No se pudo cargar la imagen {ruta_imagen}")
        return None

    gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
    gris_resized = cv2.resize(gris, (100, 100))

    return gris_resized.flatten()