import cv2
import numpy as np

import Back.preparacion_imagenes as prep

# cv2.face solo existe si está instalado 'opencv-contrib-python'
# (el paquete 'opencv-python' normal NO lo trae).
try:
    cv2.face
except AttributeError as e:
    raise ImportError(
        "cv2.face no está disponible. El reconocimiento LBPH requiere "
        "'opencv-contrib-python' en vez de (o además de) 'opencv-python'.\n"
        "Solución: pip uninstall opencv-python && pip install opencv-contrib-python"
    ) from e


# Distancia LBPH a partir de la cual se considera que YA NO es la misma
# persona (más bajo = más parecido). Valores típicos: 50 (estricto) a 100
# (permisivo). Ajustar según pruebas reales con tus fotos.
UMBRAL_DISTANCIA_LBPH = 80.0


def entrada(ruta_original, ruta_rec, array_nombres):
    """
    Construye la lista de imágenes de entrenamiento (X) y sus nombres (y)
    a partir de las fotos originales y las recortadas.
    """
    X = []
    y = []

    for nombre in array_nombres:

        # Imagen original
        img1 = prep.preparacion_img(ruta_original + nombre)
        if img1 is None:
            print(f"[entrada] Se omite (original) porque no se pudo procesar: {nombre}")
        else:
            X.append(img1)
            y.append(nombre)

        # Imagen recortada
        img2 = prep.preparacion_img(ruta_rec + nombre)
        if img2 is None:
            print(f"[entrada] Se omite (recortada) porque no se pudo procesar: {nombre}")
        else:
            X.append(img2)
            y.append(nombre)

    return X, y


def entrenar(x, y):
    """
    Entrena un reconocedor facial LBPH (Local Binary Patterns Histograms):
    el algoritmo de OpenCV diseñado específicamente para reconocimiento de
    rostros, en vez del KNN anterior sobre imágenes binarizadas (que no
    distinguía rasgos faciales reales).

    Devuelve un diccionario con el modelo entrenado y el mapeo id -> nombre,
    porque LBPH exige etiquetas numéricas, no strings.
    """
    if len(x) == 0:
        raise ValueError(
            "No hay datos de entrenamiento válidos (todas las imágenes "
            "fallaron al procesarse)."
        )

    nombres_unicos = sorted(set(y))
    nombre_a_id = {nombre: i for i, nombre in enumerate(nombres_unicos)}
    id_a_nombre = {i: nombre for nombre, i in nombre_a_id.items()}

    etiquetas = np.array([nombre_a_id[nombre] for nombre in y])

    reconocedor = cv2.face.LBPHFaceRecognizer_create()
    reconocedor.train(x, etiquetas)

    return {
        "modelo": reconocedor,
        "id_a_nombre": id_a_nombre,
    }


def consultar(clf, x):
    """
    Consulta el modelo entrenado con una lista de imágenes ya preparadas
    (normalmente una sola: [img_test]).

    Devuelve (nombres_predichos, confianza) donde confianza está en escala
    0 a 1 (más alto = más parecido), para mantener la misma interfaz que
    usaba vista.py con el KNN anterior (comparación contra 0.7).
    """
    if any(elemento is None for elemento in x):
        raise ValueError("consultar() recibió una imagen no válida (None).")

    modelo = clf["modelo"]
    id_a_nombre = clf["id_a_nombre"]

    nombres_predichos = []
    confianzas = []

    for img in x:
        id_predicho, distancia = modelo.predict(img)

        # LBPH da una DISTANCIA (0 = idéntico, mientras más alto peor
        # coincide). La convertimos a una "confianza" 0-1 para no tener
        # que tocar la lógica de vista.py.
        confianza = max(0.0, 1.0 - (distancia / UMBRAL_DISTANCIA_LBPH))

        nombres_predichos.append(id_a_nombre[id_predicho])
        confianzas.append(confianza)

    return nombres_predichos, max(confianzas)
