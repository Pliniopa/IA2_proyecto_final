from sklearn.neighbors import KNeighborsClassifier
import numpy as np
import Back.preparacion_imagenes as prep

def entrada(ruta_original, ruta_rec, array_nombres):
    X = []
    y = []
    for nombre in array_nombres:
        img1 = prep.preparacion_img(ruta_original + nombre)
        if img1 is not None:
            X.append(img1)
            y.append(nombre)

        img2 = prep.preparacion_img(ruta_rec + nombre)
        if img2 is not None:
            X.append(img2)
            y.append(nombre)

    return np.array(X), np.array(y)

def entrenar(x, y):
    n_vecinos = min(7, len(x))
    clf = KNeighborsClassifier(n_neighbors=max(1, n_vecinos))
    clf = clf.fit(x, y)
    return clf

def consultar(clf, x):
    x = np.array(x)
    if x.ndim == 1:
        x = x.reshape(1, -1)
        
    pred = clf.predict(x)
    prob = clf.predict_proba(x)
    return pred, prob.max()