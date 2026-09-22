import cv2


def preparacion_img(imagen_a_Procesar):
    imagen_origin = cv2.imread(imagen_a_Procesar)
    if imagen_origin is None:
        print("Imagenes no encontradas")
    else:
        imagen_gris = cv2.cvtColor(imagen_origin, cv2.COLOR_BGR2GRAY) # Paso 1 Pipeline: Conversión a escala de grises
        _, imagen_binaria = cv2.threshold(imagen_gris, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)# Paso 2 Pipeline: Umbralización (Otsu o Binaria Invertida según el fondo) Asumiendo objetos oscuros sobre fondo claro
        

        # Paso 3 Pipeline: Limpieza Morfológica (Apertura para eliminar ruido pequeño y separar contornos)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        imagen_limpia = cv2.morphologyEx(imagen_binaria, cv2.MORPH_OPEN, kernel)

        # Paso 4 Pipeline: Detección de Contornos
        contornos, _ = cv2.findContours(imagen_limpia, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Lógica empresarial: Umbral de área "X" para clasificar objetos
        UMBRAL_AREA_X = 3000  # Píxeles (Ajustar según la resolución de tu imagen)

        print("--- RESULTADOS DE CLASIFICACIÓN DE OBJETOS ---")

        for i, cnt in enumerate(contornos, start=1):
            # Paso 3 (Requisito): Calcular área
            area = cv2.contourArea(cnt)
    
            # Ignorar ruido insignificante
            if area < 100:
                continue

         # Imprimir área en consola
            print(f"Objeto #{i}: Área = {area:.2f} px²")

            # Bounding box
            x, y, w, h = cv2.boundingRect(cnt)

            # Paso 4 (Requisito): Lógica empresarial
            if area > UMBRAL_AREA_X:
             # Objeto grande -> Bounding Box AZUL en BGR: (255, 0, 0)
                color_box = (255, 0, 0)
                etiqueta = "Grande"
            else:
                # Objeto pequeño -> Bounding Box ROJO en BGR: (0, 0, 255)
                color_box = (0, 0, 255)
                etiqueta = "Pequeno"

            # Dibujar Bounding Box y etiqueta
            cv2.rectangle(imagen_origin, (x, y), (x + w, y + h), color_box, 2)
            cv2.putText(imagen_origin, f"{etiqueta} ({int(area)})", (x, y - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_box, 1)

        imagen_limpia = cv2.resize( imagen_limpia, (100, 100)) #Redimensiona la imagen
        return imagen_limpia.flatten() #aplanado de imagen para salida y almacenaje







