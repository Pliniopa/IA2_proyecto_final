from PIL import Image

def recorte(ruta, array):
    ruta_almacen=f"./Proyecto/Recursos/imagenes_entr_recortadas/{array}"
    # 1. Cargar la imagen original
    imagen = Image.open(ruta)
    texto = str(array)
    resta = 300
    resta2 = 200

    # 2. Definir las coordenadas del cuadro: (izquierda, arriba, derecha, abajo)
    # Ejemplo: recortar un cuadro de 300x300 desde la posición (100, 50)
   
    izquierda = 846 - resta2
    arriba = 861 

    #izquierda = 846 - resta2
    #arriba = 761 - resta
    #derecha = izquierda + 2591 # izquierda + ancho deseado (100 + 300)
    #abajo = arriba + 2526  # arriba + alto deseado (50 + 300)
    
    
    
    derecha = 2526
    abajo = 2591 + resta
    
    

    caja_corte = (izquierda, arriba, derecha, abajo)

    # Validación: la caja de recorte debe caber dentro de la imagen real.
    # Antes esto fallaba en silencio (PIL rellena con negro o recorta menos
    # de lo esperado) cuando una imagen tenía otra resolución.
    ancho_img, alto_img = imagen.size
    if derecha > ancho_img or abajo > alto_img or izquierda < 0 or arriba < 0:
        print(
            f"[recorte] Aviso: la caja de recorte {caja_corte} excede el "
            f"tamaño real de '{texto}' ({ancho_img}x{alto_img}). "
            "Se ajustará a los límites de la imagen."
        )
        derecha = min(derecha, ancho_img)
        abajo = min(abajo, alto_img)
        izquierda = max(izquierda, 0)
        arriba = max(arriba, 0)
        caja_corte = (izquierda, arriba, derecha, abajo)

    if derecha <= izquierda or abajo <= arriba:
        print(f"[recorte] Error: caja de recorte inválida para '{texto}': {caja_corte}. Se omite.")
        return

    # 3. Recortar la imagen
    imagen_recortada = imagen.crop(caja_corte)

    # 4. Guardar el resultado
    imagen_recortada.save(ruta_almacen)
    #imagen_recortada.show()