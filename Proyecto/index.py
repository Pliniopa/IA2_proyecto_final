import Back.nombre_img as imagenes
import Back.recorte_imagen as rec
import Back.preparacion_imagenes as prep


ruta = ("./Proyecto/Recursos/Imagenes_entrenamiento/")

array_nombres = imagenes.obtener_nombres_jpg(ruta)
imagenes_preparadas = []


#print(array_nombres)
for i in range(len(array_nombres)):
    ruta_img = ruta + array_nombres[i]
    rec.recorte(ruta_img, array_nombres[i])
    imagen_preparada = prep.preparacion_img(ruta_img, array_nombres[i])
    imagenes_preparadas.append(imagen_preparada)
    
print(imagenes_preparadas[0])