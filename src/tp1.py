# -*- coding: utf-8 -*-

import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

# Ejercicio 1: Lectura, visualización y escritura de imágenes.


# 1. Realice la carga y visualiación de diferentes imágenes.
def ej1_1():
    img1 = cv.imread("camino.tif")  # RGB
    img2 = cv.imread("botellas.tif")  # RGB

    cv.imshow("Camino", img1)
    cv.imshow("Botellas", img2)
    cv.waitKey(0)
    cv.destroyAllWindows()


# 2. Muestre en pantalla información sobre las imágenes.
def ej1_2():
    img1 = cv.imread("camino.tif")
    print("filas: %d columnas: %d canales: %d" % (img1.shape[0], img1.shape[1], img1.shape[2]))
    print("valor medio %.3f, mínimo %d, máximo %d" % (np.mean(img1), np.min(img1), np.max(img1)))


# 3. Investigue la forma en que se almacena la imagen y como leer y como escribir
# un valor puntual de la imagen (vea Mat.at ).
def ej1_3():
    img1 = cv.imread("camino.tif")
    print img1

    img1[1:20, 1:20, :] = 0
    cv.imshow("Camino", img1)
    cv.waitKey(0)
    cv.destroyAllWindows()


# 4. Defina y recorte una subimagen de una imagen (vea ROI, Region Of Interest).
def ej1_4():
    img1 = cv.imread("camino.tif")
    img2 = img1[1:img1.shape[0]/2, 1:img1.shape[1]/2, :]
    cv.imshow("Camino", img1)
    cv.waitKey(0)
    cv.imshow("Camino", img2)
    cv.waitKey(0)
    cv.destroyAllWindows()

# 5. Investigue y realice una función que le permita mostrar varias imágenes en
# una sóla ventana.
def ej1_5():
    image = cv.imread('camino.tif')

    grey = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    # Make the grey scale image have three channels
    grey_3_channel = cv.cvtColor(grey, cv.COLOR_GRAY2BGR)

    # Otra forma de hacer lo mismo: np.hstack
    # numpy_horizontal = np.hstack((image, grey_3_channel))

    numpy_horizontal_concat = np.concatenate((image, grey_3_channel), axis=1)

    cv.imshow('Numpy Horizontal Concat', numpy_horizontal_concat)
    cv.waitKey()

# Ejercicio 2: Información de intensidad.
# 1. Informe los valores de intensidad de puntos particulares de la imagen (opcio-
# nal: determine la posición en base al click del mouse).

def ej2_1():
    img1 = cv.imread("camino.tif", cv.IMREAD_GRAYSCALE)
    print (img1[5, 5])

    line = 200
    plt.figure()
    plt.imshow(img1, cmap='gray')
    plt.plot([0, img1.shape[1]], [line, line])

    plt.figure()
    plt.plot(img1[line, :])
    plt.title("Perfil de intensidad")

    plt.figure()
    # cv.calcHist(images, channels, mask, histSize, ranges[, hist[, accumulate]])
    hist = cv.calcHist([img1], [0], None, [256], [0, 256])
    plt.bar(range(256), np.squeeze(hist))
    plt.title("Histograma de la imagen")

    plt.show()


def ej3():
    img1 = cv.imread("botellas.tif", cv.IMREAD_GRAYSCALE)
    line = img1.shape[0]/2
    plt.figure()


    # contamos donde la intensidad es cero en una linea horizontal
    perfilh = img1[line, :]
    ceros = [0]
    for index, valor in enumerate(perfilh):
        if valor == 0:
            ceros.append(index)
    ceros.append(img1.shape[1])
    centros = []

    # calculamos los centros entre dos espacios negros
    for i, val in enumerate(ceros):
        if i == 0:
            continue
        else:
            if ceros[i] != (ceros[i-1] + 1):
                centros.append((ceros[i] + ceros[i-1])/2)

    vectorcito = np.zeros(256, dtype=int)
    for intensidad in range(256):
        if intensidad < 200:
            vectorcito[intensidad] = 0
        else:
            vectorcito[intensidad] = 1

    # img2 va a tener un umbral en 200
    img2 = cv.LUT(img1, vectorcito)
    plt.axis("off")
    plt.imshow(img2, cmap='gray')

    # eliminamos primer conjunto de ceros
    for pos, val in enumerate(centros):
        # print img2[:, val], "\n"
        botella = img2[:, val]
        for pos2, i in enumerate(botella):
            if i == 1:
                break
        botella = botella[pos2:botella.shape[0]-2]

        ceros = 0
        for i in botella:
            if i == 0:
                ceros += 1

        porcentaje_lleno = ceros / float(botella.shape[0])

        if porcentaje_lleno*100 < 80:
            print "La botella ", pos+1, " está llena al ",porcentaje_lleno*100,"%\n"
            plt.plot([val, val], [0, img1.shape[0]], color="red")
        # graficar los centros

    plt.show()


# 2. Obtenga y grafique los valores de intensidad (perfil de intensidad ) sobre una
# determinada fila o columna.
# 3. Grafique el perfil de intensidad para un segmento de interés cualquiera.

if __name__ == '__main__':
    ej3()
