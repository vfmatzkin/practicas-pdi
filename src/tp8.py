# -*- coding: utf-8 -*-
import scipy

import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
import funciones as func
import time


#  1. Implemente y/o utilice las funciones de alguna libreria para realizar erosion
# y dilatacion en imagenes binarias.
#  Aplique ambas operaciones a imagenes reales binarizadas. Adicione diferentes
# cantidades de ruido sal y pimienta a estas imagenes y vuelva a efectuar las
# operaciones. Revise los resultados.
def ej1():
    img = cv.imread('../img/textobin.png', cv.IMREAD_GRAYSCALE)
    func.graficar(img, 255, 0, 'gray', 'Imagen cargada')
    ret, img = cv.threshold(img, 128, 255, cv.THRESH_BINARY)

    # Erosionar con un kernel cuadrado
    kernelcuadrado = np.ones((3,3),np.uint8)
    erosion1 = cv.erode(img,kernelcuadrado, iterations = 1)
    func.graficar(erosion1, 255, 0, 'gray', 'Imagen erosionada con cuadrado')

    # Erosionar con un kernel cruz
    kernelcruz = cv.getStructuringElement(cv.MORPH_CROSS, (3, 3))
    erosion2 = cv.erode(img,kernelcruz, iterations = 3)
    func.graficar(erosion2, 255, 0, 'gray', 'Imagen erosionada con cruz')

    # Dilatar con un kernel cruz 3 veces
    kernelcruz = cv.getStructuringElement(cv.MORPH_CROSS, (3, 3))
    dilata1 = cv.dilate(erosion2,kernelcruz, iterations = 3)
    func.graficar(dilata1, 255, 0, 'gray', 'Imagen erosionada con cruz 3 veces y dilatada 3 veces con cruz')

    # Ensuciar con ruido sal y pimienta
    salp = func.ruidoSalPimienta(img, 0.5, 0.05)
    func.graficar(salp, 255, 0, 'gray', 'Imagen ensuciada')

    # Erosionar con un kernel cruz la imagen sucia
    erosion3 = cv.erode(img,kernelcruz, iterations = 1)
    func.graficar(erosion3, 255, 0, 'gray', 'Imagen sucia (SyP) erosionada con cruz')


# 2. Implemente, reutilizando las funciones anteriores, funciones para realizar las
# operaciones de apertura y cierre.
#  Realice ambos procesos sobre la imagen que se presenta a continuación con
# el elemento estructurante proporcionado. Generar otros elementos estructu-
# rantes, realizar las operaciones y comentar los resultados.
def ej2():
    img = cv.imread('../img/ej2.png', cv.IMREAD_GRAYSCALE)
    func.graficar(img, 255, 0, 'gray', 'Imagen cargada')
    ret, img = cv.threshold(img, 128, 255, cv.THRESH_BINARY)

    # Kernel que voy a usar para ambas operaciones
    kernelcuadrado = np.ones((3,3),np.uint8)

    # func.apertura con un kernel cuadrado
    salida = func.apertura(img, kernelcuadrado)
    func.graficar(salida, 255, 0, 'gray', 'Operacion func.apertura')

    # func.cierre con un kernel cuadrado
    salida = func.cierre(img, kernelcuadrado)
    func.graficar(salida, 255, 0, 'gray', 'Operacion func.cierre')

    # Probando otros elementos estructurantes
    kernelcruz = cv.getStructuringElement(cv.MORPH_CROSS, (3, 3))
    kernelelipse = cv.getStructuringElement(cv.MORPH_ELLIPSE, (3, 3))

    salidacruzap = func.apertura(img, kernelcruz)
    salidaelipseap = func.apertura(img, kernelelipse)

    func.graficar(salidacruzap, 255, 0, 'gray', 'Operacion func.apertura con cruz')
    func.graficar(salidaelipseap, 255, 0, 'gray', 'Operacion func.apertura con elipse')

    salidacruzci = func.cierre(img, kernelcruz)
    salidaelipseci = func.cierre(img, kernelelipse)

    func.graficar(salidacruzci, 255, 0, 'gray', 'Operacion func.cierre con cruz')
    func.graficar(salidaelipseci, 255, 0, 'gray', 'Operacion func.cierre con elipse')


#  3.1. Obtenga el nombre completo, profesión y las siglas de la empresa a la que
# pertenece la tarjeta personal de la imagen 'Tarjeta.jpeg'

# Esto se puede mejorar usando el resultado dilatado (y más limpio) como máscara
# para la imagen de entrada.
def ej3_1():
    img = cv.imread('../img/Tarjeta.jpeg', cv.IMREAD_GRAYSCALE)
    img = cv.resize(img, (2*img.shape[1], 2*img.shape[0]))

    ret, imgb = cv.threshold(img, 128, 255, cv.THRESH_BINARY)
    imgb = imgb.max() - imgb
    func.graficar(imgb, 255, 0, 'gray', 'Imagen cargada (binarizada en 128)')

    kernelcruz = cv.getStructuringElement(cv.MORPH_CROSS, (3,3))

    # Erosionar con un kernel cruz
    erosion1 = cv.erode(imgb, kernelcruz, iterations=1)
    # func.graficar(erosion1, 255, 0, 'gray', 'Imagen erosionada con cruz')

    # Dilatar con un kernel cruz
    dilat1 = cv.dilate(erosion1, kernelcruz, iterations=1)
    # func.graficar(dilat1, 255, 0, 'gray', 'Imagen dilatada con cruz')

    # Erosionar con un kernel cruz
    erosion2 = cv.erode(dilat1, kernelcruz, iterations=1)
    # func.graficar(erosion2, 255, 0, 'gray', 'Imagen erosionada con cruz 2')

    median = cv.medianBlur(erosion2, 5)
    # func.graficar(median, 255, 0, 'gray', 'Mediana')

    # Dilatar con un kernel cruz
    dilat2 = cv.dilate(median, kernelcruz, iterations=1)
    # func.graficar(dilat2, 255, 0, 'gray', 'Imagen dilatada 2 con cruz')

    imgf = dilat2

    imgf = imgf.max() - imgf
    func.graficar(imgf, 255, 0, 'gray', 'Imagen final')


# 3.2 Usando la imagen Caracteres.jpeg Extraiga solo las letras.
# Luego extraiga toodo simbolo que sea diferente a letras.
def ej3_2():
    img = cv.imread('../img/Caracteres.jpeg', cv.IMREAD_GRAYSCALE)
    img = cv.resize(img, (2*img.shape[1], 2*img.shape[0]))

    ret, imgb = cv.threshold(img, 128, 255, cv.THRESH_BINARY)
    imgb = imgb.max() - imgb
    func.graficar(img, 255, 0, 'gray', 'Imagen cargada (binarizada en 128)')

    kernelcruz = cv.getStructuringElement(cv.MORPH_CROSS, (3,3))

    # Erosionar con un kernel cruz
    erosion1 = cv.erode(imgb, kernelcruz, iterations=1)
    # func.graficar(erosion1, 255, 0, 'gray', 'Imagen erosionada con cruz')

    # Dilatar con un kernel cruz
    dilat1 = cv.dilate(erosion1, kernelcruz, iterations=1)
    # func.graficar(dilat1, 255, 0, 'gray', 'Imagen dilatada con cruz')

    # Erosionar con un kernel cruz
    erosion2 = cv.erode(dilat1, kernelcruz, iterations=1)
    # func.graficar(erosion2, 255, 0, 'gray', 'Imagen erosionada con cruz 2')

    median = cv.medianBlur(erosion2, 5)
    # func.graficar(median, 255, 0, 'gray', 'Mediana')

    # Erosionar con un kernel cuadrado
    kernelcuadrado = np.ones((3,3),np.uint8)
    erosion2 = cv.dilate(median,kernelcuadrado, iterations = 1)
    # func.graficar(erosion2, 255, 0, 'gray', 'Imagen erosionada con cuadrado')

    imgf = erosion2

    imgf = imgf.max() - imgf
    func.graficar(imgf, 255, 0, 'gray', 'Imagen final')

    # Hasta acá se extraen las letras grandes..
    # Ahora se extraen los simbolos

    simbolos = img * imgf
    erosion3 = cv.erode(simbolos, kernelcruz, iterations=1)
    func.graficar(erosion3, 255, 0, 'gray', 'Simbolos')


# 3.3  Umbralice la imagen de las estrellas y extraiga de la misma sólo las estrellas
# que se observen de mayor tamaño
def ej3_3():
    img = cv.imread('../img/estrellas.jpg', cv.IMREAD_GRAYSCALE)
    # imgBin = cv.adaptiveThreshold(img, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 11, 2)
    ret, imgBin = cv.threshold(img,100,255,cv.THRESH_BINARY)
    # ret, imgBin = cv.threshold(img,0,255,cv.THRESH_BINARY+cv.THRESH_OTSU)

    #Creo kernel cuadrado
    kernelCuad = np.ones((3,3),np.uint8)

    #Erosiono
    erosionada = cv.erode(imgBin,kernelCuad, iterations = 1)
    func.graficar(erosionada, 255, 0, 'gray', 'Erosion')

    # Enmascaro
    mascara = erosionada / erosionada.max()
    final = mascara * img

    func.graficar(img, 255, 0, 'gray', 'Imagen original')
    func.graficar(final, 255, 0, 'gray', 'Imagen Final')


#  3.4. Diseñe un elemento estructurante especifico para poder extraer la estrella
# fugaz de la imagen lluviaEstrellas.jpg
def ej3_4():
    img = cv.imread('../img/lluviaEstrellas.jpg', cv.IMREAD_GRAYSCALE)
    img = cv.resize(img, (2*img.shape[1], 2*img.shape[0]))

    # Binarizar la imagen
    ret, thres = cv.threshold(img, 25, 255, cv.THRESH_BINARY)

    # Elemento estructurante
    structElem = np.array([[0, 0, 1],[0, 1, 0],[1, 0, 0]], dtype=np.uint8)
    kernelcuadrado = np.ones((3,3),np.uint8)

    # Operaciones morfológicas
    ero1 = cv.erode(thres, structElem, iterations=7)

    # La máscara tiene que tener valores 0 o 1
    mask = ero1 / 255

    # Multiplicar la mascara por la imagen original para extraer la estrella
    imgf = mask * img

    # func.graficar
    func.graficar(img, 255, 0, 'gray', 'Imagen original')
    func.graficar(thres, 255, 0, 'gray', 'Imagen umbralizada')
    func.graficar(ero1, 255, 0, 'gray', 'Imagen erosionada')
    func.graficar(imgf, 255, 0, 'gray', 'Estrella fugaz extraida')


#  3.5 De la imagen que se presenta a continuación, usted debe eliminar todos aque-
# llos glóbulos rojos que estén en contacto (directo o indirecto) con el borde:
def ej3_5():
    img = cv.imread('../img/Globulos Rojos.jpg', cv.IMREAD_GRAYSCALE)
    ret, imgBin = cv.threshold(img, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)

    imgBin = imgBin.max() - imgBin
    elemEstruc = np.ones((3, 3), np.uint8)

    #Le saco los bordes a la original
    imgBordes = imgBin.copy()
    for i in range(2, imgBordes.shape[0] - 2):
        for j in range(2, imgBordes.shape[1] - 2):
            imgBordes[i, j] = 0

    # Reconstruccion morfologica
    #Ciclo, se repite hasta que la img anterio sea igual a la actual
    imgBordesAnterior = np.zeros((imgBordes.shape[0], imgBordes.shape[1]), np.uint8)

    while ((imgBordes != imgBordesAnterior).max()):
        imgBordesAnterior = imgBordes.copy()
        #Diltar
        dilatacion = cv.dilate(imgBordesAnterior, elemEstruc, iterations=1)

        #And
        imgBordes = np.bitwise_and(dilatacion, imgBin)

    # LLENADO DE CENTROS
    imgBordesL = imgBin.copy()
    imgBordesL = imgBordesL.max() - imgBordesL
    for i in range(2, imgBordes.shape[0] - 2):
        for j in range(2, imgBordes.shape[1] - 2):
            imgBordesL[i, j] = 0

    imgBordesAnterior = np.zeros((imgBordes.shape[0], imgBordes.shape[1]), np.uint8)

    while ((imgBordesL != imgBordesAnterior).max()):
        imgBordesAnterior = imgBordesL.copy()
        #Diltar
        dilatacion = cv.dilate(imgBordesAnterior, elemEstruc, iterations=1)

        #And
        imgBordesL = np.bitwise_and(dilatacion, (imgBordes.max() - imgBordes))

    imgFinal = imgBordesL.max() - imgBordesL

    func.graficar(imgFinal, 255, 0, 'gray', 'Imagen Binaria')

    #Aplico la mascara
    imgFinal = imgFinal.max() - imgFinal
    imgFinal = imgFinal / imgFinal.max()

    Resultado = imgFinal * img

    # Graficar
    # func.graficar(img, 255, 0, 'gray', 'Imagen original')
    # func.graficar(imgBin, 255, 0, 'gray', 'Imagen Binaria')
    # func.graficar(imgBordes, 255, 0, 'gray', 'Imagen Binaria Solo Bordes')
    # func.graficar(imgFinal, 255, 0, 'gray', 'Imagen Binaria Llenado')
    # func.graficar(Resultado, 255, 0, 'gray', 'Resultado')

# 3.6 Utilizando la imagen satelital que se presenta a continuación,
# genere una máscara binaria con el rio de la Plata y sus mayores afluentes
# utilizando la máscara obtenga la información de la imagen original
# obtenga el contorno de los rios
def ej3_6():
    img = cv.imread('../img/Rio.jpeg')
    rows, cols = img.shape[0:2]
    # ret, imgBin = cv.threshold(img, 114, 255, cv.THRESH_BINARY)

    img_hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    H = img_hsv[:, :, 0]
    S = img_hsv[:, :, 1]
    V = img_hsv[:, :, 2]

    # La idea es ir segmentando por partes (y luego sumar esas partes), empezando por
    # lo más fácil, que es el río. Su H está alrededor de 112.
    h_rio, tol1 = 112, 25
    ret, mask_rio1 = cv.threshold(H, h_rio - tol1, 255, cv.THRESH_BINARY)
    mask_rio1 = cv.medianBlur(mask_rio1, 9)
    mask_rio1 = cv.blur(mask_rio1, (15,15))
    ret, mask_rio1 = cv.threshold(mask_rio1, 85, 255, cv.THRESH_BINARY)

    # Ahora saco los afluentes, para ello primero umbralizo en S. Luego veo que zonas grandes
    # quedaron y comparo con la informacion de la mascara 1, ya que las zonas grandes deben tener
    # un color similar.
    s_aflu = 128
    ret, mask_rio2 = cv.threshold(S, s_aflu, 255, cv.THRESH_BINARY)
    mask_rio2 = 255 - mask_rio2

    # Obtengo de esta mascara las zonas grandes (mediana) y mediante la resta con la mascara 1
    mask_rio3 = cv.medianBlur(mask_rio2, 47)
    mask_rio4 = mask_rio3 - mask_rio1

    # La diferencia entre la máscara 3 y la 4, es que la 3 contiene partes del río que sí
    # corresponden, mientras que la 4 tiene zonas homogeneas grandes que no corresponden.
    ret, mask_rio4 = cv.threshold(mask_rio4, 200, 255, cv.THRESH_BINARY)

    mask_rio4 = (255 - mask_rio4)/255

    # Elimino de la máscara 2 lo que no va:
    mask_rio2 = mask_rio2 * mask_rio4

    # Componentes conectadas de la mascara 2
    output = cv.connectedComponentsWithStats(mask_rio2, 4, cv.CV_32S)
    labels = output[1]

    # Tengo muchas componentes conectadas, como 5000, por lo tanto, ubico cual es la que me
    # interesa y al resto la anulo o las dejo como estan
    labels[np.where(labels > 208)] = 0

    # Para poder umbralizar tengo que cambiar el tipo de entero
    labels = labels.astype(np.uint8)

    # umbralizo en el label que me interesa
    ret, mask_rio2 = cv.threshold(labels, 190, 255, cv.THRESH_BINARY)

    # sumo las mascaras
    mask_rio = mask_rio1 + mask_rio2

    # Para multiplicarla, convierto los valores distintos de cero a 1
    mask_rio[np.where(mask_rio != 0)] = 1

    # Dilato para conectar un poco
    mask_rio = cv.dilate(mask_rio, np.ones(5, dtype=np.uint8), iterations=3)

    # Aplico la mascara a la imagen original
    solorio = img.copy()
    solorio[:,:,0] = img[:,:,0] * mask_rio
    solorio[:,:,1] = img[:,:,1] * mask_rio
    solorio[:,:,2] = img[:,:,2] * mask_rio

    # todo mejorar: Aplicar desenfoque a la máscara, eliminar partes de la mascara que sean G en la img

    func.graficar(img[:,:,::-1], 255, 0, 'gray', 'Imagen de entrada')
    # func.graficar(H, 255, 0, 'gray', 'Canal H')
    # func.graficar(S, 255, 0, 'gray', 'Canal S')
    # func.graficar(V, 255, 0, 'gray', 'Canal V')
    # func.graficar(mask_rio1, 255, 0, 'gray', 'Mascara 1')
    # func.graficar(mask_rio2, 255, 0, 'gray', 'Mascara 2')
    # func.graficar(mask_rio3, 255, 0, 'gray', 'Mascara 3')
    # func.graficar(mask_rio4, 1, 0, 'gray', 'Mascara 4')
    # func.graficar(mask_rio, 255, 0, 'gray', 'Suma')
    func.graficar(solorio[:,:,::-1], 255, 0, 'gray', 'Solo rio')

#  3.7 Encuentre la envoltura convexa del melanoma de la imagen Melanoma.jpg
def ej3_7v2():
    img = cv.imread('../img/Melanoma.jpg', cv.IMREAD_GRAYSCALE)
    img_o = cv.imread('../img/Melanoma.jpg')
    ret, imgBin = cv.threshold(img, 170, 255, cv.THRESH_BINARY)

    # Filtro pasa bajos
    n = 15
    img_fpb = cv.blur(img, (n, n))

    # Umbral
    ret, umbral = cv.threshold(img_fpb, 164, 255, cv.THRESH_BINARY)

    ret_img, contours, hierarchy = cv.findContours(umbral, 2, 1)
    cnt = contours[0]
    hull = cv.convexHull(cnt, returnPoints = False)
    defects = cv.convexityDefects(cnt, hull)

    # imgn = cv.drawContours(img_o, contours, -1, [255, 0, 0])
    func.graficar(img_o[:,:,::-1], 255, 0, 'gray', 'Contorno')
    for i in range(defects.shape[0]):
        s, e, f, d = defects[i, 0]
        start = tuple(cnt[s][0])
        end = tuple(cnt[e][0])
        plt.plot(start, end, 'ro-')
        # cv.line(img, start, end, [0, 255, 0], 2)
    plt.figure()
    # points = np.transpose(np.where(umbral))
    # hull = scipy.spatial.ConvexHull(points)

    # Graficar
    func.graficar(img, 255, 0, 'gray', 'Imagen de entrada')
    func.graficar(umbral, 255, 0, 'gray', 'Imagen umbralizada')
    # func.graficar(bordes, 255, 0, 'gray', 'Bordes')
    # func.graficar(hull, hull.max(), 0, 'gray', 'Convex Hull')


def ej3_7():
    img = cv.imread('../img/Melanoma.jpg', cv.IMREAD_GRAYSCALE)
    ret, imgBin = cv.threshold(img, 170, 255, cv.THRESH_BINARY)
    imgBin = cv.medianBlur(imgBin, 31)
    imgBin = imgBin.max() - imgBin
    # #imgBin = cv.adaptiveThreshold(img, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 11, 2)
    #
    #Elementos estructurantes
    B1 = np.array((
        [1, 0, 0],
        [1, 0, 0],
        [1, 0, 0]), dtype="uint8")

    B2 = np.array((
        [1, 1, 1],
        [0, 0, 0],
        [0, 0, 0]), dtype="uint8")

    B3 = np.array((
        [0, 0, 1],
        [0, 0, 1],
        [0, 0, 1]), dtype="uint8")

    B4 = np.array((
        [0, 0, 0],
        [0, 0, 0],
        [1, 1, 1]), dtype="uint8")
    #
    # # FUNCION IMPORTANTEEEEEEE!!! GUARDAR BIEN (HIT OR MISS)
    # # cv.morphologyEx(input_image, cv.MORPH_HITMISS, kernel)
    #
    imgAnterior = np.zeros((imgBin.shape[0],imgBin.shape[1]), np.uint8)
    convexHull = imgBin.copy()

    while ((convexHull != imgAnterior).max()):
        imgAnterior = convexHull.copy()

        x1 = cv.morphologyEx(convexHull, cv.MORPH_HITMISS, B1)
        x2 = cv.morphologyEx(convexHull, cv.MORPH_HITMISS, B2)
        x3 = cv.morphologyEx(convexHull, cv.MORPH_HITMISS, B3)
        x4 = cv.morphologyEx(convexHull, cv.MORPH_HITMISS, B4)

        convexHull = np.bitwise_or(convexHull, x1)
        convexHull = np.bitwise_or(convexHull, x2)
        convexHull = np.bitwise_or(convexHull, x3)
        convexHull = np.bitwise_or(convexHull, x4)

    func.graficar(imgBin, 255, 0, 'gray', 'Imagen original')
    func.graficar(convexHull, convexHull.max(), 0, 'gray', 'Convex Hull')


#  3.8 Encuentre los esqueletos de los cuerpos presentes en la imagen Cuerpos.jg
def ej3_8():
    img = cv.imread('../img/Cuerpos.jpg', cv.IMREAD_GRAYSCALE)
    ret, imgBin = cv.threshold(img, 200, 255, cv.THRESH_BINARY)
    imgBin = imgBin.max() - imgBin

    #Dilato para llenar los huecos
    B = np.ones((3, 3), np.uint8)
    dilatacion = cv.dilate(imgBin, B, iterations=4)

    #Erociono para que vuelva al tamaño original
    erosion = cv.erode(dilatacion, B, iterations=4)

    #Ahora a sacar esqueletosss
    Sk = np.zeros((erosion.shape[0], erosion.shape[1]), np.uint8)
    Paso1 = cv.erode(erosion, B, iterations=1)

    while (Paso1.max() != 0):
        Paso2 = func.cierre(Paso1, B)
        Sk = Sk + (Paso1 - Paso2)

        Paso1 = cv.erode(Paso1, B, iterations=1)

    func.graficar(img, 255, 0, 'gray', 'Imagen de entrada')
    func.graficar(imgBin, 255, 0, 'gray', 'Imagen umbralizada')
    func.graficar(erosion, 255, 0, 'gray', 'Dilatada')
    func.graficar(Sk, 255, 0, 'gray', 'Skeleton')


if __name__ == '__main__':
    ej3_5()
    plt.show()