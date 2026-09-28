"""
============================================================================
 RECONOCIMIENTO FACIAL DE ALTA PRECISIÓN - SOLO PYTHON (SIN DLIB / SIN C++)
============================================================================
Librería utilizada: ÚNICAMENTE OpenCV (opencv-contrib-python)

MEJORA respecto a la versión anterior (Haar Cascade + LBPH):
    Esta versión usa dos modelos de redes neuronales modernos, ya
    integrados en OpenCV, mucho más precisos y robustos frente a
    cambios de ángulo, distancia e iluminación:

    1. YuNet  (cv2.FaceDetectorYN)
       -> Detección de rostros mediante una red neuronal ligera.
          Mucho más precisa que Haar Cascade, especialmente con
          rostros en ángulo, parcialmente girados o a distancia.

    2. SFace  (cv2.FaceRecognizerSF)
       -> Calcula un "embedding" (vector de 128 características) por
          rostro, igual de robusto sea cual sea la pose o la luz.
          Se comparan embeddings con similitud coseno.

    Ambos son modelos ONNX pre-entrenados que se cargan directamente
    con OpenCV: NO requieren compilar nada, solo descargar 2 archivos
    de pesos (ver instrucciones más abajo / en el README).

Estructura esperada junto a este script:
    face_detection_yunet_2023mar.onnx
    face_recognition_sface_2021dec.onnx
    rostros_conocidos/
        APELLIDO_NOMBRE.jpg
        APELLIDO_NOMBRE_2.jpg   (opcional: más fotos = más precisión)
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

# Umbral de similitud coseno (SFace). Va de -1 a 1: MÁS ALTO = MÁS PARECIDO
# (al revés que LBPH). El valor recomendado oficialmente por OpenCV Zoo es
# 0.363 (calibrado para muy pocos falsos positivos). Puedes subirlo si
# quieres ser más estricto, o bajarlo un poco si no te reconoce.
UMBRAL_SIMILITUD = 0.363

INDICE_CAMARA = 0
EXTENSIONES_VALIDAS = (".jpg", ".jpeg", ".png")


# ----------------------------------------------------------------------
# 2. VERIFICACIÓN DE ARCHIVOS NECESARIOS
# ----------------------------------------------------------------------
def verificar_modelos():
    """Confirma que los 2 archivos .onnx existen antes de continuar."""
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
# 3. FUNCIONES AUXILIARES
# ----------------------------------------------------------------------
def nombre_desde_archivo(nombre_archivo: str) -> str:
    """'Silvia_Aguilar_2.jpg' -> 'Silvia Aguilar'."""
    base = os.path.splitext(nombre_archivo)[0]
    base = re.sub(r"_\d+$", "", base)
    nombre = base.replace("_", " ").strip()
    return " ".join(p.capitalize() for p in nombre.split())


def obtener_embedding(imagen_bgr, detector, reconocedor):

    alto, ancho = imagen_bgr.shape[:2]

    if ancho > 1280:
        escala = 1280 / ancho
        imagen_bgr = cv2.resize(
            imagen_bgr,
            (
                int(ancho * escala),
                int(alto * escala)
            )
        )

    alto, ancho = imagen_bgr.shape[:2]

    detector.setInputSize((ancho, alto))
    _, rostros = detector.detect(imagen_bgr)

    if rostros is None or len(rostros) == 0:
        return None

    rostro_principal = max(
        rostros,
        key=lambda r: r[2] * r[3]
    )

    rostro_alineado = reconocedor.alignCrop(
        imagen_bgr,
        rostro_principal
    )

    embedding = reconocedor.feature(rostro_alineado)

    return embedding

def construir_base_de_datos(carpeta: str, detector, reconocedor):
    """
    Recorre todas las imágenes de 'carpeta', calcula el embedding de cada
    rostro y las agrupa por persona (según nombre de archivo). Si una
    persona tiene varias fotos, se promedian sus embeddings para mayor
    precisión.

    Devuelve: lista de (nombre_persona, embedding_promedio)
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

    print(f"[INFO] Procesando {len(archivos)} imágenes de referencia...")

    embeddings_por_persona = {}   # nombre -> lista de embeddings
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

        nombre = nombre_desde_archivo(archivo)
        embeddings_por_persona.setdefault(nombre, []).append(embedding)

    if fotos_sin_rostro:
        print(f"[ADVERTENCIA] No se detectó rostro en {len(fotos_sin_rostro)} foto(s): "
              f"{', '.join(fotos_sin_rostro)}")

    if not embeddings_por_persona:
        print("[ERROR] No se pudo procesar ninguna foto (ningún rostro detectado).")
        sys.exit(1)

    base_datos = []
    for nombre, lista_embeddings in embeddings_por_persona.items():
        # Promedia los embeddings si hay varias fotos de la misma persona.
        promedio = np.mean(np.array(lista_embeddings), axis=0)
        base_datos.append((nombre, promedio))
        print(f"       - {nombre}: {len(lista_embeddings)} foto(s)")

    print(f"[INFO] Base de datos lista con {len(base_datos)} persona(s).")
    return base_datos


def identificar_mejor_coincidencia(embedding_actual, base_datos, reconocedor):
    """
    Compara 'embedding_actual' contra todas las personas de la base de
    datos usando similitud coseno y devuelve (nombre, similitud) de la
    mejor coincidencia.
    """
    mejor_nombre = None
    mejor_similitud = -1.0

    for nombre, embedding_referencia in base_datos:
        similitud = reconocedor.match(
            embedding_actual, embedding_referencia, cv2.FaceRecognizerSF_FR_COSINE
        )
        if similitud > mejor_similitud:
            mejor_similitud = similitud
            mejor_nombre = nombre

    return mejor_nombre, mejor_similitud


# ----------------------------------------------------------------------
# 4. FUNCIÓN PRINCIPAL: CAPTURA EN VIVO Y RECONOCIMIENTO
# ----------------------------------------------------------------------
def ejecutar_reconocimiento_en_vivo(base_datos, detector, reconocedor, url_camera):
    captura = cv2.VideoCapture(url_camera)
    captura.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    captura.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)



    if not captura.isOpened():
        print(f"[ERROR] No fue posible acceder a la cámara (índice {INDICE_CAMARA}).")
        sys.exit(1)

    print("[INFO] Cámara iniciada. Presiona 'q' para salir.")

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
                    # rostro = [x, y, w, h, --5 puntos de landmarks--, score]
                    x, y, w, h = rostro[0:4].astype(int)

                    rostro_alineado = reconocedor.alignCrop(frame, rostro)
                    embedding_actual = reconocedor.feature(rostro_alineado)

                    nombre_detectado, similitud = identificar_mejor_coincidencia(
                        embedding_actual, base_datos, reconocedor
                    )

                    # DEBUG: útil para calibrar UMBRAL_SIMILITUD.
                    print(f"[DEBUG] mejor coincidencia = {nombre_detectado}  "
                          f"similitud = {similitud:.3f}  (umbral = {UMBRAL_SIMILITUD})")

                    if similitud >= UMBRAL_SIMILITUD:
                        etiqueta = f"{nombre_detectado} ({similitud:.2f})"
                        color = (0, 255, 0)  # Verde -> coincidencia
                    else:
                        etiqueta = f"Desconocido ({similitud:.2f})"
                        color = (0, 0, 255)  # Rojo -> sin coincidencia

                    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                    cv2.rectangle(frame, (x, y + h - 25), (x + w, y + h), color, cv2.FILLED)
                    cv2.putText(
                        frame, etiqueta, (x + 6, y + h - 6),
                        cv2.FONT_HERSHEY_DUPLEX, 0.55, (255, 255, 255), 1,
                    )

            cv2.imshow(
                "Reconocimiento Facial de Alta Precisión - Presiona 'q' para salir", frame
            )

            if cv2.waitKey(1) & 0xFF == ord("q"):
                print("[INFO] Tecla 'q' detectada. Cerrando programa...")
                break

    finally:
        captura.release()
        cv2.destroyAllWindows()
        print("[INFO] Recursos liberados. Programa finalizado.")


# ----------------------------------------------------------------------
# 5. PUNTO DE ENTRADA DEL PROGRAMA
# ----------------------------------------------------------------------
if __name__ == "__main__":
    verificar_modelos()

    detector = cv2.FaceDetectorYN.create(
        RUTA_MODELO_DETECCION, "", (320, 320),
        score_threshold=0.6, nms_threshold=0.3, top_k=5000, # sensibilidad
    )
    reconocedor = cv2.FaceRecognizerSF.create(RUTA_MODELO_RECONOCIMIENTO, "")

    base_datos = construir_base_de_datos(CARPETA_ROSTROS_CONOCIDOS, detector, reconocedor)

    url = input("Ingrese url Camara: ")

    ejecutar_reconocimiento_en_vivo(base_datos, detector, reconocedor, url)
