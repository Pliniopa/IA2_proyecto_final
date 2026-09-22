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

    # 3. Recortar la imagen
    imagen_recortada = imagen.crop(caja_corte)

    
    # 4. Guardar o mostrar el resultado
    
    
    imagen_recortada.save(ruta_almacen)
    #imagen_recortada.show()