import Back.nombre_img as imagenes
import Back.recorte_imagen as rec
import Back.preparacion_imagenes as prep


ruta = "./Proyecto/Recursos/Imagenes_entrenamiento/"
ruta_rec = "./Proyecto/Recursos/imagenes_entr_recortadas/"

array_nombres = imagenes.obtener_nombres_jpg(ruta)
imagenes_preparadas = []


def img_prep(array_nombres, ruta):
    for i in range(len(array_nombres)):
        ruta_img = ruta + array_nombres[i]
        rec.recorte(ruta_img, array_nombres[i])
    return print("fin")



ciclo = True

while ciclo == True:
    print("Seleccione una opción:")
    print("1. Recortar imágenes")
    print("2. Salir")
    opcion = input("Ingrese el número de la opción deseada: ")
    match opcion:
        case "1":
            img_prep(array_nombres, ruta)
            ciclo = False
        case "2":
            ciclo = False
        case _:
            print("Ingrese un valor valido")
            input()

#print(array_nombres)
for i in range(len(array_nombres)):
    ruta_img = ruta_rec + array_nombres[i]
    imagen_preparada = prep.preparacion_img(ruta_img)
    imagenes_preparadas.append(imagen_preparada)
    
print(imagenes_preparadas[0])