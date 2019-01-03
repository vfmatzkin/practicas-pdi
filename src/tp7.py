# -*- coding: utf-8 -*-

import cv2 as cv
from matplotlib import pyplot as plt
import numpy as np
from click_and_crop import *
import funciones as func

# 1.1 Escriba una función que implemente el detector de bordes de Roberts. La
# función debe obtener como salida una imagen binaria conteniendo los bordes
# detectados. Aplíquelo sobre la imagen ’estanbul.tif’.

# 1.2 Incorpore a la función anterior los detectores de bordes de Prewitt, Sobel,
# Laplaciano y LoG, permitiendo al usuario seleccionar cualquiera de ellos.
# Compare los resultados obtenidos con los diferentes métodos.

# 1.3 Estudie el uso del detector de bordes de Canny provisto por la librería opencv,
# evalue los resultados de variar los parámetros. Compare los resultados obtenidos
# con los métodos previos.
def ej1():
    img = cv.imread("../img/estanbul.tif", cv.IMREAD_GRAYSCALE)
    img = cv.imread("../img/letras1.tif", cv.IMREAD_GRAYSCALE)
    func.graficar(img, 255, 0, 'gray', 'Imagen de entrada')

    umbral1, umbral2 = 128, 255

    robertsxy = func.bordes_roberts(img, umbral1, umbral2)
    prewittxy = func.bordes_prewitt(img, umbral1, umbral2)
    sobelxy = func.bordes_sobel(img, umbral1, umbral2)
    laplacianoxy = func.bordes_laplaciano(img, umbral1, umbral2)
    laplacianologxy = func.bordes_LoG(img, umbral1, umbral2)

    # 1.1 Roberts
    func.graficar(robertsxy, 255, 0, 'gray', 'Roberts')

    # 1.2 Otros algoritmos
    func.graficar(prewittxy, 255, 0, 'gray', 'Prewitt')
    func.graficar(sobelxy, 255, 0, 'gray', 'Sobel')
    func.graficar(laplacianoxy, 255, 0, 'gray', 'Laplaciano')
    func.graficar(laplacianologxy, 255, 0, 'gray', 'Laplaciano del Gaussiano')

    # 1.3 Canny
    umbral1 = 150
    umbral2 = 200
    cannyxy = cv.Canny(img,umbral1,umbral2)
    func.graficar(cannyxy, 255, 0, 'gray', 'Canny')


# 1.4: Cargue la imagen ’mosquito.jpg’ y genere a partir de ella versiones con ruido
# de tipo gaussiano con media cero y distintos valores de desvío estándar. Aplique
# los distintos operadores en cada caso y compare su desempeño.
#
# Solución: Al tener más ruido la detección de bordes empeora.
def ej1_4():
    img = cv.imread("../img/mosquito.jpg", cv.IMREAD_GRAYSCALE)
    func.graficar(img, 255, 0, 'gray', 'Imagen de entrada')

    ru1 = func.sumar_ruido_gauss(img, 0, 10)
    ru2 = func.sumar_ruido_gauss(img, 0, 15)
    ru3 = func.sumar_ruido_gauss(img, 0, 29)

    umbral1 = 150
    umbral2 = 200

    func.graficar(ru1, 255, 0, 'gray', 'Imagen de entrada con ruido gaussiano (media: 0, desvio: 10)')
    func.graficar(ru2, 255, 0, 'gray', 'Imagen de entrada con ruido gaussiano (media: 0, desvio: 15)')
    func.graficar(ru3, 255, 0, 'gray', 'Imagen de entrada con ruido gaussiano (media: 0, desvio: 29)')


    # robertsxy = bordes_roberts(img, umbral1, umbral2)
    # prewittxy = bordes_prewitt(img, umbral1, umbral2)
    sobelxy1 = func.bordes_sobel(ru1, umbral1, umbral2)
    sobelxy2 = func.bordes_sobel(ru2, umbral1, umbral2)
    sobelxy3 = func.bordes_sobel(ru3, umbral1, umbral2)
    # laplacianoxy = bordes_laplaciano(img, umbral1, umbral2)
    # laplacianologxy = bordes_LoG(img, umbral1, umbral2)
    #
    # # 1.1 Roberts
    # graficar(robertsxy, 255, 0, 'gray', 'Roberts')
    #
    # # 1.2 Otros algoritmos
    # graficar(prewittxy, 255, 0, 'gray', 'Prewitt')
    func.graficar(sobelxy1, 255, 0, 'gray', 'Sobel')
    func.graficar(sobelxy2, 255, 0, 'gray', 'Sobel')
    func.graficar(sobelxy3, 255, 0, 'gray', 'Sobel')
    # graficar(laplacianoxy, 255, 0, 'gray', 'Laplaciano')
    # graficar(laplacianologxy, 255, 0, 'gray', 'Laplaciano del Gaussiano')
    #
    # # 1.3 Canny
    # cannyxy = cv.Canny(img,umbral1,umbral2)
    # graficar(cannyxy, 255, 0, 'gray', 'Canny')


# 2.3 Estudie los métodos HoughLines y HoughLinesP provistos por OpenCV, explique sus diferencias,
# ventajas y desventajas.
def ej2():
    """
    cv2.HoughLines():
    First parameter, Input image should be a binary image, so apply threshold or use canny edge
    detection before finding applying hough transform. Second and third parameters are \rho and \theta
    accuracies respectively. Fourth argument is the threshold, which means minimum vote it should get
    for it to be considered as a line.
    """
    img_o = cv.imread("../img/letras1.tif")
    img_p = img_o.copy()
    img = cv.cvtColor(img_o,cv.COLOR_BGR2GRAY)
    func.graficar(img, 255, 0, 'gray', 'Bordes')

    bordes = cv.Canny(img, 20, 30)
    func.graficar(bordes, 255, 0, 'gray', 'Bordes')

    lines = cv.HoughLines(bordes, 1, np.pi / 180, 90)
    for linea in lines:
        for rho, theta in linea:
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a * rho
            y0 = b * rho
            x1 = int(x0 + 1000 * (-b))
            y1 = int(y0 + 1000 * (a))
            x2 = int(x0 - 1000 * (-b))
            y2 = int(y0 - 1000 * (a))

            cv.line(img_o, (x1, y1), (x2, y2), (0, 0, 255), 2)

    func.graficar(img_o[:,:,::-1], 255, 0, 'gray', 'Hough')

    """
        cv2.HoughLinesP():
        In the hough transform, you can see that even for a line with two arguments, it takes a lot of computation. 
        Probabilistic Hough Transform is an optimization of Hough Transform we saw. It doesn’t take all the points into 
        consideration, instead take only a random subset of points and that is sufficient for line detection. Just we 
        have to decrease the threshold. See below image which compare Hough Transform and Probabilistic Hough Transform 
        in hough space. 
    """
    minLineLength = 100
    maxLineGap = 10
    lines = cv.HoughLinesP(bordes, 1, np.pi / 180, 100, minLineLength, maxLineGap)
    for line in lines:
        cv2.line(img_p, (line[0][0], line[0][1]), (line[0][2], line[0][3]), (0, 0, 255), 6)

    func.graficar(img_p[:, :, ::-1], 255, 0, 'gray', 'HoughP')


def capturarMouse(event, x, y, flag, param):
    global pt
    if event == cv.EVENT_LBUTTONDOWN:
        llenar_hueso(x,y)


def llenar_hueso(x,y):
    img = cv.imread("../img/bone.tif", 0)
    (m, n) = img.shape

    mask = np.zeros((m + 2, n + 2), np.uint8)

    img_c = img.copy()
    lo = 10
    up = 50
    new_color = (255, 255, 255)
    cv.floodFill(img_c, mask, (x, y), new_color, lo, up)

    cv.imshow('Resultado', img_c)

    cv.waitKey()


#  3. Segmentación mediante crecimiento de regiones
#  Cargue la imagen ’bone.tif’. Segmente algún área de interés en la imagen uti-
# lizando el método de crecimiento de regiones, a partir de una semilla seleccionada
# por el usuario (puede hacerlo mediante un click o por consola).
def ej3():
    """
    Cargue la imagen ’bone.tif’. Segmente algún área de interés en la imagen utilizando el método de crecimiento de
    regiones, a partir de una semilla seleccionada por el usuario (puede hacerlo mediante un click o por consola).
    """
    img = cv.imread("../img/bone.tif", 0)

    cv.namedWindow('image')
    cv.imshow('image', img)

    cv.setMouseCallback('image', capturarMouse)

    cv.waitKey()


# Ejercicio 4: Segmentación en color y etiquetado
# El objetivo del ejercicio es poder identificar las rosas presentes en una imagen de un
# ramo de flores para poder contarlas, procesarlas por separado, compararlas, etc.
# 1. Cargue la imagen ’rosas.jpg’. A partir de los métodos de segmentación en
# color vistos en el Trabajo Práctico 3, obtenga una imagen binaria con las rosas
# segmentadas.
# 2. Descarte las regiones erróneas con el método que considere apropiado (mor-
# fológico, regla ad-hoc, filtro de suavizado y binarización, etc). Según el método
# elegido este paso puede ser posterior al etiquetado.
# 3. Identifique las diferentes regiones por el método de etiquetado de componentes
# conectadas. ¿Podría realizar la misma operación utilizando el algoritmo de crec-
# imiento de regiones? Pruébelo y saque conclusiones.
# 4. Cuente automáticamente la cantidad de rosas presente en la imagen original.
# Sobre la imagen original, dibuje un círculo de tamaño arbitrario en el centro de
# cada rosa (varíe la opacidad si lo considera necesario).
def ej4():
    img = cv.imread("../img/rosas.jpg")

    imgHSV = cv.cvtColor(img, cv.COLOR_BGR2HSV)

    #PASO 1 (segmentar)

    #Quiero segmentar el rojo (Angulo 0)

    # Centroides de H y S
    medH = 180 #Media de H = 0 = 180 por opencv
    medS = 200

    # Calculo los radios de  H y S como la varianza de cada componente
    rH = 15
    rS = 50

    parteH = imgHSV[:, :, 0]
    parteS = imgHSV[:, :, 1]
    parteV = imgHSV[:, :, 2]

    # Recorro imgCompleta y veo si esta dentro de los circulos en cada canal
    imgSegmendata = np.zeros((parteH.shape[0], parteS.shape[1], 3), dtype=np.uint8)

    for i in range(parteH.shape[0]):
        for j in range(parteS.shape[1]):
            if parteH[i, j] <= medH + rH and parteH[i, j] >= medH - rH:
                if parteS[i, j] <= medS + rS and parteS[i, j] >= medS - rS:
                    imgSegmendata[i, j, 0] = parteH[i, j]
                    imgSegmendata[i, j, 1] = parteS[i, j]
                    imgSegmendata[i, j, 2] = parteV[i, j]



    #graficar(imgHSV[:, :, 0], 255, 0, 'gray', 'Solo H')

    imgHSV = cv.cvtColor(imgHSV, cv.COLOR_HSV2BGR)
    func.graficar(imgHSV[:, :, ::-1], 255, 0, 'gray', 'imagen original')


    imgSegmendata = cv.cvtColor(imgSegmendata, cv.COLOR_HSV2BGR)
    func.graficar(imgSegmendata[:, :, ::-1], 255, 0, 'gray', 'imagen Segmentada')


    # PASO 2 (Descartar regiones erroneas, hacer homogena y demas)

    imgBN = cv.cvtColor(imgSegmendata, cv.COLOR_BGR2GRAY)
    ret, imgBN = cv.threshold(imgBN, 10, 255, cv.THRESH_BINARY)

    #Quiero hacer homogenea la region

    imgBN = cv.blur(imgBN , (15,15)) #Desenfoco
    ret, imgBN = cv.threshold(imgBN, 127, 255, cv.THRESH_BINARY) #Binarizo de nuevo

    #Dilato
    elemStruct = np.ones((3,3))
    imgBN = cv.dilate(imgBN, elemStruct)

    #PASO 3 (Contar cuantas rosas hay)

    #Hago componentes conectadas

    # You need to choose 4 or 8 for connectivity type
    connectivity = 4
    # Perform the operation
    output = cv2.connectedComponentsWithStats(imgBN, connectivity, cv2.CV_32S)
    # Get the results
    # The first cell is the number of labels
    num_labels = output[0]
    # The second cell is the label matrix
    labels = output[1] #Imagen donde cada componente conectada tiene un valor de intensidad diferente
    # The third cell is the stat matrix
    stats = output[2]
    # The fourth cell is the centroid matrix
    centroids = output[3] #EL CENTROIDE[0,0] ES EL DEL FONDO!!!!

    print 'CANTIDAD DE ROSAS = ', num_labels - 1

    func.graficar(imgBN, 255, 0, 'gray', 'imagen ByN')
    func.graficar(labels, labels.max(), 0, 'gray', 'Componentes conectadas')

    #PASO 4 (Dibujar circulos sobre las rosas)

    ret_image, contours, hierarchy = cv2.findContours(imgBN, 2, 1)

    cnt = contours[0]
    M = cv2.moments(cnt)

    radius = 12
    for i in range(num_labels-1):
        center = (int(centroids[i+1, 0]), int(centroids[i+1, 1]))
        cv2.circle(img, center, radius, (0, 255, 0), 2)


    func.graficar(img[:,:,::-1], 255, 0, 'gray', 'Imagen Final')

# Componentes conectadas!
# Labels is a matrix the size of the input image where each element has a value equal to its label.
#
# Stats is a matrix of the stats that the function calculates. It has a length equal to the number of labels and a width equal to the number of stats. It can be used with the OpenCV documentation for it:
#
#     Statistics output for each label, including the background label, see below for available statistics. Statistics are accessed via stats[label, COLUMN] where available columns are defined below.
#
#         cv2.CC_STAT_LEFT The leftmost (x) coordinate which is the inclusive start of the bounding box in the horizontal direction.
#         cv2.CC_STAT_TOP The topmost (y) coordinate which is the inclusive start of the bounding box in the vertical direction.
#         cv2.CC_STAT_WIDTH The horizontal size of the bounding box
#         cv2.CC_STAT_HEIGHT The vertical size of the bounding box
#         cv2.CC_STAT_AREA The total area (in pixels) of the connected component
#
# Centroids is a matrix with the x and y locations of each centroid. The row in this matrix corresponds to the label number.


# 5. Implemente un código para segmentar en forma automática la pista de ater-
# rizaje principal en las imágenes de aeropuertos (corrientes ruidogris.jpg e
# iguazu ruidogris.jpg), las cuales poseen una combinación de ruido gaussiano
# e impulsivo. La salida del proceso debe ser la imagen restaurada con la pista
# principal coloreada (por ejemplo, con rectas rojas).
# Realice un proceso general, no adaptado a las particularidades de las imágenes
# de prueba (por ejemplo, la localización, el largo o la inclinación de la pista).
# Para probar esta característica, se le sugiere que genere imágenes rotadas y/o
# desplazadas de las propuestas.
def ej5():
    """
    Implemente un código para segmentar en forma automática la pista de aterrizaje principal en las imágenes de
    aeropuertos (corrientes ruidogris.jpg e iguazu ruidogris.jpg), las cuales poseen una combinación de ruido
    gaussiano e impulsivo. La salida del proceso debe ser la imagen restaurada con la pista principal coloreada
    (por ejemplo, con rectas rojas).
    Realice un proceso general, no adaptado a las particularidades de las imágenes de prueba (por ejemplo, la
    localización, el largo o la inclinación de la pista). Para probar esta caracteristica, se le sugiere que
    genere imágenes rotadas y/o desplazadas de las propuestas.
    """
    # Cargamos las imágenes en escala de grises
    img1 = cv.imread("../img/corrientes_ruidogris2.jpg", cv.IMREAD_GRAYSCALE)
    img2 = cv.imread("../img/iguazu_ruidogris2.jpg", cv.IMREAD_GRAYSCALE)

    # Subimágenes
    subimgC = sub_image("../img/corrientes_ruidogris2.jpg")
    subimgI = sub_image("../img/iguazu_ruidogris2.jpg")

    # Eliminación del ruido impulsivo: Mediana de 3x3
    # Se realiza en la imagen y en la subimagen, para analizar el histograma
    medianaC = cv.medianBlur(img1, 3)
    medianasC = cv.medianBlur(subimgC, 3)

    medianaI = cv.medianBlur(img2, 3)
    medianasI = cv.medianBlur(subimgI, 3)

    # Filtro adaptativo: Para calcularlo debo pasarle una varianza estimada, se usa
    # la varianza de la zona homogenea. El filtro se aplica a la imagen y a la zona
    # homogenea para ver los cambios.
    varianzaC = np.std(medianasC)
    varianzaI = np.std(medianasI)

    print "Varianzas mediana Corrientes, Iguazu: ",varianzaC, varianzaI

    fasC = func.filter_adaptative2(medianasC[:,:,0], varianzaC,5)
    faC = func.filter_adaptative2(medianaC, varianzaC,5)

    fasI = func.filter_adaptative2(medianasI[:,:,0], varianzaI,5)
    faI = func.filter_adaptative2(medianaI, varianzaI,5)

    varianzamC = np.std(fasC)
    varianzamI = np.std(fasI)

    print "Varianzas mediana Corrientes, Iguazu luego de Filtro adaptativo: ", varianzamC, varianzamI

    # Para dibujar líneas a color debo tener imágenes a color.
    img_o1 = cv.cvtColor(faC, cv.COLOR_GRAY2RGB)
    img_o2 = cv.cvtColor(faI, cv.COLOR_GRAY2RGB)

    # Detectado de bordes
    umbral1, umbral2 = 130, 255
    canny1 = cv.Canny(faC,umbral1,umbral2)

    umbral1, umbral2 = 100, 255
    canny2 = cv.Canny(faI,umbral1,umbral2)

    img_o3 = cv.cvtColor(canny1, cv.COLOR_GRAY2RGB)
    img_o4 = cv.cvtColor(canny2, cv.COLOR_GRAY2RGB)

    # Transformada de Hough
    linesC = cv.HoughLines(canny1, 1, np.pi / 180, 90)
    # Gráfica de la primera línea detectada (la más fuerte)
    for rho, theta in linesC[0]:
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho
        x1 = int(x0 + 1000 * (-b))
        y1 = int(y0 + 1000 * (a))
        x2 = int(x0 - 1000 * (-b))
        y2 = int(y0 - 1000 * (a))

        cv.line(img_o1, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv.line(img_o3, (x1, y1), (x2, y2), (0, 0, 255), 2)


    linesI = cv.HoughLines(canny2, 1, np.pi / 180, 30)
    for rho, theta in linesI[0]:
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho
        x1 = int(x0 + 1000 * (-b))
        y1 = int(y0 + 1000 * (a))
        x2 = int(x0 - 1000 * (-b))
        y2 = int(y0 - 1000 * (a))

        cv.line(img_o2, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv.line(img_o4, (x1, y1), (x2, y2), (0, 0, 255), 2)


    # Gráficas
    func.graficar(img1, 255, 0, 'gray', 'Imagen de entrada 1')
    func.graficar(img2, 255, 0, 'gray', 'Imagen de entrada 2')
    func.histograma(subimgC, "Histograma de una zona homogenea de Corrientes")
    func.histograma(subimgI, "Histograma de una zona homogenea de Iguazu")
    func.histograma(medianasC, "Histograma de una zona homogenea de Corrientes luego de mediana")
    func.histograma(medianasI, "Histograma de una zona homogenea de Iguazu luego de mediana")
    func.graficar(medianaC, 255, 0, 'gray', 'Mediana de Corrientes')
    func.graficar(medianaI, 255, 0, 'gray', 'Mediana de Iguazu')
    func.histograma(fasC, "Histograma de una zona homogenea de Corrientes luego de mediana y Filtro Adaptativo")
    func.histograma(fasI, "Histograma de una zona homogenea de Iguazu luego de mediana y Filtro Adaptativo")
    func.graficar(faC, 255, 0, 'gray', 'Imagen de Corrientes despues del adaptativo')
    func.graficar(faI, 255, 0, 'gray', 'Imagen de Iguazu despues del adaptativo')
    func.graficar(canny1, 255, 0, 'gray', 'Canny de Corrientes')
    func.graficar(canny2, 255, 0, 'gray', 'Canny de Iguazu')
    func.graficar(img_o1[:, :, ::-1], 255, 0, 'gray', 'Hough de Corrientes')
    func.graficar(img_o2[:, :, ::-1], 255, 0, 'gray', 'Hough de Iguazu')
    func.graficar(img_o3[:, :, ::-1], 255, 0, 'gray', 'Hough de Canny de Corrientes')
    func.graficar(img_o4[:, :, ::-1], 255, 0, 'gray', 'Hough de Canny de Iguazu')


if __name__ == '__main__':
    ej4()
    plt.show()
