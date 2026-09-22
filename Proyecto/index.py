import Back.nombre_img as imagenes
import Back.recorte_imagen as rec
import Back.neuronas as neuronas
import Front.vista as vista
import Back.preparacion_imagenes as prep

ruta = "./Proyecto/Recursos/Imagenes_entrenamiento/" # Imagenes Originales
ruta_rec = "./Proyecto/Recursos/imagenes_entr_recortadas/" # Imagenes recortadas


img_probar = "./Proyecto/Front/Almacenaje"
array_nombres_2 = imagenes.obtener_nombres_jpg(img_probar) #Obtencion de Nombres de la carpeta de almacenamiento

array_nombres = imagenes.obtener_nombres_jpg(ruta) #Obtencion de Nombres de la carpeta original
#imagenes_preparadas = []
#imagen_origin = []
#img_origin = []

#for nombre in array_nombres:
    #imagen_origin.append(ruta + str(nombre))

#for j in range (len(imagen_origin)):
    #img_origin.append(prep.preparacion_img(imagen_origin[j])) #Imagenes Originales

def img_prep(array_nombres, ruta): # funcion para recortar imagenes
    for i in range(len(array_nombres)):
        ruta_img = ruta + array_nombres[i]
        rec.recorte(ruta_img, array_nombres[i])
    return print("fin")



ciclo = True

while ciclo == True: # Opciones para recortar imagenes (usado para cuando se agrueguen mas datos)
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
#for i in range(len(array_nombres)): #Preparacion de imagen (sesion 6)
    #ruta_img = ruta_rec + array_nombres[i]
    #imagen_preparada = prep.preparacion_img(ruta_img)
    #imagenes_preparadas.append(imagen_preparada)
    
#print(imagenes_preparadas[0])




xsalida,ysalida = neuronas.entrada(
    ruta,ruta_rec,array_nombres) #Entrenamiento de neuronas


result = neuronas.entrenar(xsalida, ysalida)


img = prep.preparacion_img( ruta_rec + "PARRA_PLINIO.jpg")
print(neuronas.consultar(result, [img]))

vista.crear_interfaz(result) # lanzamiento de interfaz
