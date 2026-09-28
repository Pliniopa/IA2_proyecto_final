# IA2_proyecto_final

-Nombre del proyecto: Reconocimiento facial mediante herramienta IA
-Breve descripción del proyecto: es una tecnología biométrica que analiza, identifica o verifica la identidad de una persona a partir de los rasgos únicos de su rostro en una imagen o video.
-carrera: ing software, 5to semestre, Jornada Nocturna
- universidad: Universitaria de colombia
-nombre completo de cada integrante 
    Silvia Fernanda Aguilar Hernandez
    Zuley Gomez
    Aura aponte
    Gustavo Holguin
    Plinio  Parra




============================================================
 PROYECTO v2: RECONOCIMIENTO FACIAL DE ALTA PRECISIÓN
 (YuNet + SFace, solo OpenCV, sin dlib / sin C++)
============================================================

QUÉ CAMBIÓ RESPECTO A LA VERSIÓN ANTERIOR (Haar + LBPH)
-----------------------------------------------------------
- Detección de rostros: ahora con YuNet (red neuronal), mucho más
  precisa que Haar Cascade, especialmente con ángulos, giros y
  distancias variables.
- Reconocimiento: ahora con SFace (embeddings), mucho más robusto que
  LBPH frente a cambios de luz y pose (justo el problema que tenías).

ESTRUCTURA DE ESTA CARPETA
---------------------------
proyecto_reconocimiento_facial_v2/
    reconocimiento_facial_alta_precision.py   <- programa principal
    capturar_foto_referencia.py               <- herramienta opcional
    rostros_conocidos/                        <- pon aquí tus 26 fotos
    face_detection_yunet_2023mar.onnx         <- debes descargarlo (paso 2)
    face_recognition_sface_2021dec.onnx       <- debes descargarlo (paso 2)

PASOS PARA DEJARLO FUNCIONANDO
--------------------------------
1) Instala las dependencias (una sola vez). La versión de
   opencv-contrib-python debe ser 4.5.4 o superior (las más recientes
   ya cumplen esto):
       python -m pip install --upgrade opencv-contrib-python numpy

2) Descarga estos DOS archivos de modelo (son solo "pesos" entrenados,
   NO requieren compilar nada) y colócalos DENTRO de esta misma carpeta,
   junto al .py (no dentro de rostros_conocidos):

   a) Detección de rostros (YuNet):
      https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx

   b) Reconocimiento facial (SFace):
      https://github.com/opencv/opencv_zoo/raw/main/models/face_recognition_sface/face_recognition_sface_2021dec.onnx

   Si el navegador los abre como texto/binario en vez de descargarlos,
   usa clic derecho -> "Guardar como" en cada enlace.

3) Copia tus 26 fotos dentro de "rostros_conocidos", nombradas como
   APELLIDO_NOMBRE.jpg (igual que antes). Si quieres sumar una segunda
   foto por persona tomada con la webcam (recomendado para mayor
   precisión), usa la herramienta incluida:
       python capturar_foto_referencia.py
   Presiona 'c' para capturar y escribe el nombre cuando te lo pida;
   la guardará automáticamente como Nombre_2.jpg dentro de
   rostros_conocidos.

4) Ejecuta el programa principal:
       python reconocimiento_facial_alta_precision.py

5) Presiona 'q' para cerrar la ventana de video.

AJUSTE DE PRECISIÓN
-----------------------
UMBRAL_SIMILITUD = 0.363
    (dentro de reconocimiento_facial_alta_precision.py, cerca del inicio)

    A diferencia de la versión anterior, aquí MÁS ALTO = MÁS ESTRICTO
    (es similitud, no distancia).
    - Súbelo (ej. 0.45) si te da falsos positivos (confunde personas).
    - Bájalo (ej. 0.30) si no te reconoce ni a ti misma.

    El programa imprime en la terminal:
        [DEBUG] mejor coincidencia = Nombre  similitud = 0.XXX
    Usa esos valores reales para calibrar el umbral ideal.


============================================================
 VERSIÓN CON CLASIFICADOR DE MACHINE LEARNING (si tu tarea lo exige)
============================================================
Si te piden explícitamente usar un algoritmo de Machine Learning
(KNN, SVM, Árbol de Decisión o Perceptrón) para la clasificación, usa
en su lugar: reconocimiento_facial_ml.py

Funciona en 2 etapas:
    1) Extrae características de cada rostro con SFace (igual que la
       versión anterior).
    2) Entrena un clasificador de scikit-learn (tú eliges cuál) sobre
       esas características para decidir de quién es cada rostro.

Instalación adicional necesaria:
    python -m pip install scikit-learn

Elige el algoritmo editando esta línea al inicio del script:
    ALGORITMO_ML = "knn"   # opciones: "knn", "svm", "arbol", "perceptron"

Ejecuta igual que los anteriores:
    python reconocimiento_facial_ml.py

Notas:
    - Con solo 1 foto por persona, KNN (k=1) es el más estable de los
      cuatro. SVM y Árbol también funcionan pero se benefician más de
      tener 2-3 fotos por persona.
    - El Perceptrón clásico no entrega probabilidades; el script lo
      calibra automáticamente (CalibratedClassifierCV) para poder
      seguir detectando "Desconocido". Si tienes muy pocas fotos por
      persona, puede no calibrar bien: verás un aviso en la terminal
      si eso ocurre.
    - Usa UMBRAL_CONFIANZA_ML (0 a 1) para ajustar qué tan seguro debe
      estar el clasificador antes de aceptar una identificación, igual
      que con los umbrales anteriores.

