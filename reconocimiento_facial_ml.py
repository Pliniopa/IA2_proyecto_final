"""
============================================================================
 RECONOCIMIENTO FACIAL CON CLASIFICADOR DE MACHINE LEARNING
 (KNN / SVM / Árbol de Decisión / Perceptrón) - SOLO PYTHON, SIN C++
============================================================================
Enfoque (2 etapas, como en cualquier pipeline clásico de reconocimiento
facial con ML):

    ETAPA 1 - EXTRACCIÓN DE CARACTERÍSTICAS (features):
        Se usa SFace (red neuronal ya entrenada, incluida en OpenCV) para
        convertir cada rostro en un vector numérico de 128 características
        (su "embedding"). Esto es equivalente a lo que en un pipeline
        clásico harían técnicas como HOG o PCA/Eigenfaces: reducir la
        imagen a un vector de características manejable.

    ETAPA 2 - CLASIFICACIÓN (Machine Learning "de toda la vida"):
        Con esos vectores de características ya calculados, se entrena
        un clasificador clásico de scikit-learn a elección:
            - KNN            (K-Nearest Neighbors)
            - SVM             (Support Vector Machine)
            - Árbol de Decisión (Decision Tree)
            - Perceptrón      (Perceptron)
        Ese clasificador es el que decide, en tiempo real, a qué persona
        pertenece cada rostro nuevo.

Librerías utilizadas:
    - OpenCV (opencv-contrib-python): captura de video, detección
      (YuNet) y extracción de características (SFace).
    - scikit-learn: el clasificador de Machine Learning en sí.

Estructura esperada junto a este script:
    face_detection_yunet_2023mar.onnx
    face_recognition_sface_2021dec.onnx
    rostros_conocidos/
        APELLIDO_NOMBRE.jpg
        APELLIDO_NOMBRE_2.jpg   (opcional, mejora el entrenamiento)
        ...
============================================================================
"""

import os
import re
import sys
import cv2
import numpy as np

# ----------------------------------------------------------------------
# 1. CONFIGURACIÓN GENERAL
# ----------------------------------------------------------------------

CARPETA_ROSTROS_CONOCIDOS = "rostros_conocidos"
RUTA_MODELO_DETECCION = "face_detection_yunet_2023mar.onnx"
RUTA_MODELO_RECONOCIMIENTO = "face_recognition_sface_2021dec.onnx"

# Elige aquí el algoritmo de Machine Learning para la clasificación:
# "knn"        -> K-Nearest Neighbors (recomendado con pocas fotos por persona)
# "svm"        -> Support Vector Machine
# "arbol"      -> Árbol de Decisión (Decision Tree)
# "perceptron" -> Perceptrón
ALGORITMO_ML = "knn"

# Umbral de confianza MÍNIMA (0 a 1) para aceptar una identificación.
# Si la probabilidad de la mejor clase es menor a esto, se etiqueta
# como "Desconocido". Nota: el Perceptrón clásico no entrega
# probabilidades, así que para ese caso este umbral se ignora
# (ver comentario más abajo).
UMBRAL_CONFIANZA_ML = 0.55

INDICE_CAMARA = 0
EXTENSIONES_VALIDAS = (".jpg", ".jpeg", ".png")


# ----------------------------------------------------------------------
# 2. VERIFICACIÓN DE ARCHIVOS NECESARIOS
# ----------------------------------------------------------------------
def verificar_modelos():
    faltantes = [
        ruta for ruta in (RUTA_MODELO_DETECCION, RUTA_MODELO_RECONOCIMIENTO)
        if not os.path.isfile(ruta)
    ]
    if faltantes:
        print("[ERROR] Faltan archivos de modelo necesarios:")
        for f in faltantes:
            print(f"        - {f}")
        print("        Descárgalos (ver README.txt) y colócalos junto a este script.")
        sys.exit(1)


# ----------------------------------------------------------------------
# 3. EXTRACCIÓN DE CARACTERÍSTICAS (ETAPA 1)
# ----------------------------------------------------------------------
def nombre_desde_archivo(nombre_archivo: str) -> str:
    """'Silvia_Aguilar_2.jpg' -> 'Silvia Aguilar'."""
    base = os.path.splitext(nombre_archivo)[0]
    base = re.sub(r"_\d+$", "", base)
    nombre = base.replace("_", " ").strip()
    return " ".join(p.capitalize() for p in nombre.split())


def obtener_embedding(imagen_bgr: np.ndarray, detector, reconocedor):
    """Detecta el rostro más grande y devuelve su embedding SFace (o None)."""
    alto, ancho = imagen_bgr.shape[:2]
    detector.setInputSize((ancho, alto))
    _, rostros = detector.detect(imagen_bgr)

    if rostros is None or len(rostros) == 0:
        return None

    rostro_principal = max(rostros, key=lambda r: r[2] * r[3])
    rostro_alineado = reconocedor.alignCrop(imagen_bgr, rostro_principal)
    return reconocedor.feature(rostro_alineado).flatten()


def extraer_dataset_entrenamiento(carpeta: str, detector, reconocedor):
    """
    Recorre todas las fotos de 'carpeta' y devuelve:
        X -> lista de vectores de características (uno por foto)
        y -> lista de nombres (etiqueta de cada vector)
    Cada foto cuenta como una muestra independiente (no se promedian),
    para que el clasificador de ML tenga más ejemplos con los que
    aprender.
    """
    if not os.path.isdir(carpeta):
        print(f"[ERROR] No se encontró la carpeta '{carpeta}'.")
        sys.exit(1)

    archivos = sorted(
        f for f in os.listdir(carpeta) if f.lower().endswith(EXTENSIONES_VALIDAS)
    )
    if len(archivos) == 0:
        print(f"[ERROR] La carpeta '{carpeta}' no contiene imágenes válidas.")
        sys.exit(1)

    print(f"[INFO] Extrayendo características de {len(archivos)} imágenes...")

    X, y = [], []
    fotos_sin_rostro = []

    for archivo in archivos:
        ruta = os.path.join(carpeta, archivo)
        imagen = cv2.imread(ruta)
        if imagen is None:
            print(f"[ADVERTENCIA] No se pudo leer '{archivo}', se omite.")
            continue

        embedding = obtener_embedding(imagen, detector, reconocedor)
        if embedding is None:
            fotos_sin_rostro.append(archivo)
            continue

        X.append(embedding)
        y.append(nombre_desde_archivo(archivo))

    if fotos_sin_rostro:
        print(f"[ADVERTENCIA] No se detectó rostro en {len(fotos_sin_rostro)} foto(s): "
              f"{', '.join(fotos_sin_rostro)}")

    if not X:
        print("[ERROR] No se pudo extraer características de ninguna foto.")
        sys.exit(1)

    print(f"[INFO] Dataset listo: {len(X)} muestras de {len(set(y))} persona(s).")
    return np.array(X), np.array(y)


# ----------------------------------------------------------------------
# 4. ENTRENAMIENTO DEL CLASIFICADOR DE ML (ETAPA 2)
# ----------------------------------------------------------------------
def entrenar_clasificador(X, y, algoritmo: str):
    """
    Entrena el clasificador de scikit-learn elegido en ALGORITMO_ML
    sobre los vectores de características X y etiquetas y.
    """
    try:
        from sklearn.neighbors import KNeighborsClassifier
        from sklearn.svm import SVC
        from sklearn.tree import DecisionTreeClassifier
        from sklearn.linear_model import Perceptron
        from sklearn.calibration import CalibratedClassifierCV
    except ImportError:
        print("[ERROR] Falta instalar scikit-learn.")
        print("        Ejecuta: python -m pip install scikit-learn")
        sys.exit(1)

    # Con pocas fotos por persona, k no puede ser mayor a la cantidad
    # mínima de muestras que tenga la clase con menos fotos.
    muestras_por_clase = min(np.unique(y, return_counts=True)[1])
    k_vecinos = max(1, min(3, muestras_por_clase))

    if algoritmo == "knn":
        print(f"[INFO] Entrenando clasificador KNN (k={k_vecinos})...")
        clasificador = KNeighborsClassifier(n_neighbors=k_vecinos, metric="cosine")
        clasificador.fit(X, y)

    elif algoritmo == "svm":
        print("[INFO] Entrenando clasificador SVM (kernel lineal, con probabilidades)...")
        clasificador = SVC(kernel="linear", probability=True, C=1.0)
        clasificador.fit(X, y)

    elif algoritmo == "arbol":
        print("[INFO] Entrenando Árbol de Decisión...")
        clasificador = DecisionTreeClassifier(max_depth=10, random_state=42)
        clasificador.fit(X, y)

    elif algoritmo == "perceptron":
        print("[INFO] Entrenando Perceptrón (calibrado para obtener probabilidades)...")
        # El Perceptrón clásico no entrega probabilidades (solo la clase
        # predicha), así que lo envolvemos con CalibratedClassifierCV
        # para poder aplicar el umbral de confianza igual que con los
        # demás algoritmos y detectar "Desconocido".
        base = Perceptron(max_iter=1000, random_state=42)
        muestras_por_clase_min = min(np.unique(y, return_counts=True)[1])
        cv_folds = max(2, min(3, muestras_por_clase_min))
        try:
            clasificador = CalibratedClassifierCV(base, cv=cv_folds)
            clasificador.fit(X, y)
        except ValueError:
            print("[ADVERTENCIA] Muy pocas fotos por persona para calibrar "
                  "probabilidades; usando Perceptrón simple (sin umbral de "
                  "confianza, no distingue 'Desconocido').")
            clasificador = base
            clasificador.fit(X, y)

    else:
        print(f"[ERROR] ALGORITMO_ML='{algoritmo}' no reconocido. "
              "Usa: knn, svm, arbol o perceptron.")
        sys.exit(1)

    return clasificador


def predecir_con_confianza(clasificador, embedding):
    """
    Devuelve (nombre_predicho, confianza) para un embedding dado.
    Si el clasificador no soporta predict_proba, la confianza es 1.0
    (es decir, no se puede filtrar "Desconocido" con ese algoritmo).
    """
    embedding = embedding.reshape(1, -1)

    if hasattr(clasificador, "predict_proba"):
        probabilidades = clasificador.predict_proba(embedding)[0]
        indice_mejor = int(np.argmax(probabilidades))
        nombre = clasificador.classes_[indice_mejor]
        confianza = float(probabilidades[indice_mejor])
        return nombre, confianza
    else:
        nombre = clasificador.predict(embedding)[0]
        return nombre, 1.0  # Sin probabilidad real disponible.


# ----------------------------------------------------------------------
# 5. CAPTURA EN VIVO Y RECONOCIMIENTO
# ----------------------------------------------------------------------
def ejecutar_reconocimiento_en_vivo(clasificador, detector, reconocedor):
    captura = cv2.VideoCapture(INDICE_CAMARA)
    if not captura.isOpened():
        print(f"[ERROR] No fue posible acceder a la cámara (índice {INDICE_CAMARA}).")
        sys.exit(1)

    print("[INFO] Cámara iniciada. Presiona 'q' para salir.")
    soporta_confianza = hasattr(clasificador, "predict_proba")

    try:
        while True:
            ret, frame = captura.read()
            if not ret:
                print("[ADVERTENCIA] No se pudo leer un fotograma de la cámara.")
                break

            alto, ancho = frame.shape[:2]
            detector.setInputSize((ancho, alto))
            _, rostros = detector.detect(frame)

            if rostros is not None:
                for rostro in rostros:
                    x, y, w, h = rostro[0:4].astype(int)

                    rostro_alineado = reconocedor.alignCrop(frame, rostro)
                    embedding = reconocedor.feature(rostro_alineado).flatten()

                    nombre_predicho, confianza = predecir_con_confianza(
                        clasificador, embedding
                    )

                    print(f"[DEBUG] predicción = {nombre_predicho}  "
                          f"confianza = {confianza:.2f}")

                    if soporta_confianza and confianza < UMBRAL_CONFIANZA_ML:
                        etiqueta = f"Desconocido ({confianza:.2f})"
                        color = (0, 0, 255)
                    else:
                        etiqueta = f"{nombre_predicho} ({confianza:.2f})"
                        color = (0, 255, 0)

                    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                    cv2.rectangle(frame, (x, y + h - 25), (x + w, y + h), color, cv2.FILLED)
                    cv2.putText(
                        frame, etiqueta, (x + 6, y + h - 6),
                        cv2.FONT_HERSHEY_DUPLEX, 0.55, (255, 255, 255), 1,
                    )

            cv2.imshow(
                f"Reconocimiento Facial ({ALGORITMO_ML.upper()}) - Presiona 'q' para salir",
                frame,
            )

            if cv2.waitKey(1) & 0xFF == ord("q"):
                print("[INFO] Tecla 'q' detectada. Cerrando programa...")
                break

    finally:
        captura.release()
        cv2.destroyAllWindows()
        print("[INFO] Recursos liberados. Programa finalizado.")


# ----------------------------------------------------------------------
# 6. PUNTO DE ENTRADA DEL PROGRAMA
# ----------------------------------------------------------------------
if __name__ == "__main__":
    verificar_modelos()

    detector = cv2.FaceDetectorYN.create(
        RUTA_MODELO_DETECCION, "", (320, 320),
        score_threshold=0.6, nms_threshold=0.3, top_k=5000,
    )
    reconocedor = cv2.FaceRecognizerSF.create(RUTA_MODELO_RECONOCIMIENTO, "")

    X, y = extraer_dataset_entrenamiento(CARPETA_ROSTROS_CONOCIDOS, detector, reconocedor)
    clasificador = entrenar_clasificador(X, y, ALGORITMO_ML)

    ejecutar_reconocimiento_en_vivo(clasificador, detector, reconocedor)
