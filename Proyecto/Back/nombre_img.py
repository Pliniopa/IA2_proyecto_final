import os
import numpy as np


def obtener_nombres_jpg(ruta_carpeta):
    """
    Obtiene los nombres de todos los archivos .jpg de una carpeta 
    y los retorna en un array de NumPy.
    """
    # Filtra los archivos que terminan en .jpg o .jpeg
    nombres = [
        f for f in os.listdir(ruta_carpeta) 
        if f.lower().endswith(('.jpg', '.jpeg'))
    ]
    
    return np.array(nombres)

