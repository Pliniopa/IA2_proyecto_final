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

        X.append(img1)
        y.append(nombre)

        # Imagen recortada
        img2 = prep.preparacion_img(
            ruta_rec + nombre
        )

        X.append(img2)
        y.append(nombre)

    return X, y


def entrenar(x,y):
    clf = KNeighborsClassifier(
        n_neighbors=7
    )
    clf = clf.fit(x,y)
    return(clf)

#def consultar(clf,x):
    #return clf.predict(x)

def consultar(clf, x):
    pred = clf.predict(x)
    prob = clf.predict_proba(x)

    return pred, prob.max()