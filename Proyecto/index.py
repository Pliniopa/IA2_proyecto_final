import Back.nombre_img as imagenes

ruta = ("./Recursos/Imagenes_entrenamiento")

array_nombres = imagenes.obtener_nombres_jpg(ruta)


print(array_nombres)
print(type(array_nombres))  # <class 'numpy.ndarray'>