import cv2

# Tamaño fijo al que se normaliza cada rostro antes de entrenar/consultar.
# Debe ser el mismo tanto para las fotos de entrenamiento como para las
# capturas en vivo de la cámara (vista.py ya recorta el rostro antes de
# llamar a esta función).
TAMANO_ROSTRO = (200, 200)


def preparacion_img(imagen_a_Procesar):
    """
    Prepara una imagen de rostro para el reconocedor facial LBPH.

    Antes esta función binarizaba la imagen (blanco/negro) para un
    ejercicio de detección de "objetos", lo cual descartaba la textura
    real de la cara y hacía que el reconocimiento no pudiera distinguir
    entre personas. Ahora se conserva la imagen en escala de grises
    (con tamaño e iluminación normalizados), que es lo que LBPH necesita
    para comparar patrones faciales reales.

    Devuelve una imagen 2D en escala de grises (numpy array), o None si
    la imagen no se pudo leer.
    """
    imagen_origin = cv2.imread(imagen_a_Procesar)

    if imagen_origin is None:
        print(f"[preparacion_img] No se pudo leer la imagen: {imagen_a_Procesar}")
        return None

    # Paso 1: escala de grises (LBPH trabaja sobre grises, no color)
    imagen_gris = cv2.cvtColor(imagen_origin, cv2.COLOR_BGR2GRAY)

    # Paso 2: tamaño fijo, para que todas las caras sean comparables entre sí
    imagen_gris = cv2.resize(imagen_gris, TAMANO_ROSTRO)

    # Paso 3: ecualización de histograma -> reduce el efecto de
    # iluminación distinta entre fotos (una de las causas típicas de que
    # el reconocimiento facial falle con luz de cámara variable)
    imagen_gris = cv2.equalizeHist(imagen_gris)

    return imagen_gris
