from sklearn.neighbors import KNeighborsClassifier

import Back.preparacion_imagenes as prep

def entrada(ruta_original, ruta_rec, array_nombres):

    X = []
    y = []

    for nombre in array_nombres:

        # Imagen original
        img1 = prep.preparacion_img(
            ruta_original + nombre
        )

        if img1 is None:
            print(f"[entrada] Se omite (original) porque no se pudo procesar: {nombre}")
        else:
            X.append(img1)
            y.append(nombre)

        # Imagen recortada
        img2 = prep.preparacion_img(
            ruta_rec + nombre
        )

        if img2 is None:
            print(f"[entrada] Se omite (recortada) porque no se pudo procesar: {nombre}")
        else:
            X.append(img2)
            y.append(nombre)

    return X, y


def entrenar(x, y):
    if len(x) == 0:
        raise ValueError(
            "No hay datos de entrenamiento válidos (todas las imágenes fallaron al procesarse)."
        )

    # n_neighbors no puede superar la cantidad de muestras disponibles
    n_vecinos = min(7, len(x))
    if n_vecinos < 7:
        print(f"[entrenar] Aviso: solo hay {len(x)} muestras, usando n_neighbors={n_vecinos}")

    clf = KNeighborsClassifier(
        n_neighbors=n_vecinos
    )
    clf = clf.fit(x, y)
    return clf

#def consultar(clf,x):
    #return clf.predict(x)

def consultar(clf, x):
    if any(elemento is None for elemento in x):
        raise ValueError("consultar() recibió una imagen no válida (None).")

    pred = clf.predict(x)
    prob = clf.predict_proba(x)

    return pred, prob.max()