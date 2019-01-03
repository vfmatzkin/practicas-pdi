# -*- coding: utf-8 -*-

from matplotlib import pyplot as plt
import cv2 as cv
import numpy as np
from funciones import *
from click_and_crop import *
import math


# 1 El  archivo  ‘patron.tif’ corresponde  a  un  patron  de  colores  que  varian  por
# columnas de rojo a azul.  En este ejercicio se estudiar a la informacion que llevan
# las componentes de los diferentes modelos de color, seg un las pautas siguientes:
# 1.1 Visualice el patron junto a las componentes R, G y B y analice como var ia
# la imagen en funcion de los valores de sus planos de color.
def ej1_1a():
    img = cv.imread("../img/patron.tif")

    # img1 = convertBGR2RGB(img)

    graficar(img[:,:,::-1], 255, 0, 'gray', 'imagen original')

    graficar(img[:,:,2], 255, 0, 'gray', 'Canal R')
    graficar(img[:,:,1], 255, 0, 'gray', 'Canal G')
    graficar(img[:,:,0], 255, 0, 'gray', 'Canal B')


# 1.1b Visualice las componentes H, S e V de la imagen y modifiquelas para obtener
# un patron en RGB que cumpla con las siguientes condiciones
def ej1_1b():
    img = cv.imread("../img/patron.tif")

    img2 = cv.cvtColor(img, cv.COLOR_BGR2HSV)

    graficar(img[:,:,::-1], 255, 0, 'gray', 'imagen original')
    graficar(img2[:,:,2], 255, 0, 'gray', 'Canal V')
    graficar(img2[:,:,1], 255, 0, 'gray', 'Canal S')
    graficar(img2[:,:,0], 255, 0, 'gray', 'Canal H')

    # Componente V en maxima intensidad (para darle mas brillo a la imagen)
    img2[:, :, 2] = 255

    # todo Cambiar rojo por azul y viceversa
    img2 = np.array(img2,dtype=np.uint16)
    img2[:, :, 0] = 120 - img2[:, :, 0]
    img2[:, :, 0] = img2[:, :, 0] % 180
    img2 = np.array(img2,dtype=np.uint8)

    # Transformar imagen HSV en RGB
    img_resultado = cv.cvtColor(img2, cv.COLOR_HSV2BGR)
    graficar(img_resultado[:,:,::-1], 255, 0, 'gray', 'imagen HSV con mas brillo')


# 1.2 Asigne a cada pixel de la imagen ‘rosas.jpg’ el color complementario al original
# modificando las componentes del modelo HSV.
def ej1_2():
    img = cv.imread("../img/rosas.jpg")
    graficar(img[:,:,::-1], 255, 0, 'gray', 'imagen original')

    # Para comparar luego: inversion en RGB
    img_rgb = img.copy()
    img_rgb = 255 - img_rgb

    # RGB -> HSV
    img = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    img = img.astype(np.uint16)

    # Componente H: inversion del color
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            aux = img[i, j, 0] + 90
            aux = aux % 180
            img[i, j, 0] = aux

    # Componente V: inversion de la intensidad
    img[:,:,2] = 255 - img[:,:,2]
    img = img.astype(np.uint8)

    # Invierto para graficar
    img = cv.cvtColor(img, cv.COLOR_HSV2BGR)

    graficar(img[:,:,::-1], 255, 0, 'gray', 'imagen invertida en HSV')
    graficar(img_rgb[:,:,::-1], 255, 0, 'gray', 'imagen invertida en RGB')


# 2 Una variante del metodo rodajas de intensidad consiste en asignar un color
# especifico a un rango de grises de la imagen original. Se propone en este ejercicio
# resaltar en la imagen ‘rio.jpg’ todas las areas con acumulaciones grandes de
# agua (rio central, ramas mayores y pequeños lagos), de manera que aparezcan
# en color amarillo.
# Los pasos a seguir son:
# (a) Analice el histograma para estimar los valores de gris minimo y maximo que
# correspondan al contenido de agua.
# (b) Genere una matriz de color que contenga en cada plano una copia de la
# imagen original.
# (c) Recorra la imagen original y, en funcion de sus grises, asigne el color amarillo
# a los pixeles que estan en el rango definido, sin modificar los pixeles restantes.
# (d) Visualice la imagen resultante y ajuste el rango de grises de ser necesario.
def ej2():
    img = cv.imread("../img/rio.jpg")
    graficar(img[:, :, ::-1], 255, 0, 'gray', 'imagen original')

    plt.figure()
    plt.hist(img[:,:,0].flatten(), 255)

    # se puede ver que los valores del rio es de 0 a 10 de intensidad en escala de grises
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if img[i,j,0] < 15 and img[i,j,0] >= 0:
                img[i,j,0] = 0
                img[i,j,1] = 255
                img[i,j,2] = 255

    graficar(img[:, :, ::-1], 255, 0, 'gray', 'imagen con rio resaltado')


# 3.1 Manejo de histograma. Se tiene la imagen ‘chairs oscura.jpg’ que se observa
# con poca luminosidad. Se pide mejorar la imagen a partir de la ecualización de
# histograma, comparando los efectos de realizarla en RGB (por planos) y HSV
# (canal V).
# Cargue la imagen original ‘chairs.jpg’ (sin oscurecer) y discuta nuevamente
# los resultados.
# Procese de igual manera otras imágenes de bajo contraste (en el sitio dispone
# de ‘flowers oscura.tif’) y analice los resultados.
def ej3_1():
    img = cv.imread("../img/chairs_oscura.jpg")
    graficar(img[:, :, ::-1], 255, 0, 'gray', 'imagen oscura')

    # Ecualizacion del histograma RGB
    img[:, :, 2] = cv.equalizeHist(img[:,:,2])
    img[:, :, 1] = cv.equalizeHist(img[:,:,1])
    img[:, :, 0] = cv.equalizeHist(img[:,:,0])

    #Grafico la ecualizacion en RGB por canales
    graficar(img[:, :, ::-1], 255, 0, 'gray', 'Ecualizacion RGB')

    #Abro la imagen oscura para hacer la ecualizacion en HSV
    img2 = cv.imread("../img/chairs_oscura.jpg")
    img2 = cv.cvtColor(img2, cv.COLOR_BGR2HSV)

    #Ecualizacion de V
    img2[:, :, 2] = cv.equalizeHist(img2[:, :, 2])

    #Grafico la imagen ecualizada en HSV
    img2 = cv.cvtColor(img2, cv.COLOR_HSV2BGR)
    graficar(img2[:, :, ::-1], 255, 0, 'gray', 'Ecualizacion HSV')

    #Abro la imagen original para comparar
    img3 = cv.imread("../img/chairs.jpg")
    graficar(img3[:, :, ::-1], 255, 0, 'gray', 'imagen sin oscurecer')


#  3.2 Realce mediante acentuado. Se tiene la imagen ‘camino.tif’ que se observa
# desenfocada. Se pide mejorar la imagen aplicando un filtro pasa altos de suma
# 1. Compare los resultados de procesar la imagen en RGB y HSV.
def ej3_2():
    img = cv.imread("../img/camino.tif")
    graficar(img[:, :, ::-1], 255, 0, 'gray', 'imagen original')

    #Filtro pasa alto 3x3 con suma 1
    filtro = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])

    #Filtro la imagen RGB en cada canal
    img[:, :, 0] = cv.filter2D(img[:,:,0], -1, filtro)
    img[:, :, 1] = cv.filter2D(img[:,:,1], -1, filtro)
    img[:, :, 2] = cv.filter2D(img[:,:,2], -1, filtro)

    graficar(img[:, :, ::-1], 255, 0, 'gray', 'Filtrado en RGB')

    #Mismo proceso pero filtrando V del modelo HSV
    img2 = cv.imread("../img/camino.tif")
    img2 = cv.cvtColor(img2, cv.COLOR_BGR2HSV)

    #Aplico filtro solo a la componente V
    img2[:,:,2] = cv.filter2D(img2[:,:,2], -1, filtro)

    img2 = cv.cvtColor(img2, cv.COLOR_HSV2BGR)

    graficar(img2[:, :, ::-1], 255, 0, 'gray', 'Filtrado en HSV')


# 4.1 Segmentación basada en color
# La segmentación es un proceso que divide la imagen en regiones. Este ejercicio tiene
# por objetivo segmentar algún color en particular en una imagen en los modelos RGB
# y HSI. En cada caso deberá determinar el subdominio a segmentar.
# 1. Segmentación en RGB. En este modelo se segmenta por el método de las rodajas
# de color con un modelo esférico.
# (a) Cargue la imagen ‘futbol.jpg’.
# (b) Tome una muestra representativa del color a segmentar y calcule el centro
# de la esfera (valor medio de cada componente).
# (c) A partir del histograma de cada componente determine el radio de la esfera.
# (d) Genera la máscara binaria recorriendo la imagen y verificando la pertenencia
# de cada pixel a la esfera.
# (e) Obtenga la imagen segmentada mediante la aplicación de la máscara sobre
# la imagen original.
def ej4_1():
    #Extraigo la subimagen que quiero segmentar
    img=sub_image("../img/futbol.jpg")
    graficar(img[:, :, ::-1], 255, 0, 'gray', 'img')

    #Calculo los centroides de RGB
    medB=np.mean(img[:,:,0])
    medG=np.mean(img[:,:,1])
    medR=np.mean(img[:,:,2])

    #Muestro los histogramas de los 3 canales
    plt.figure()
    plt.hist(img[:,:,0].flatten(), 255, color='b')
    plt.hist(img[:,:,1].flatten(), 255, color='g')
    plt.hist(img[:,:,2].flatten(), 255, color='r')

    #A ojo le doy el radio para la segmentacion
    rB = np.std(img[:,:,0])
    rG = np.std(img[:,:,1])
    rR= np.std(img[:,:,2])

    print rB, rG, rR

    #Abro la imagen completa
    imgCompleta = cv.imread("../img/futbol.jpg")
    parteB = imgCompleta[:,:,0]
    parteG = imgCompleta[:,:,1]
    parteR = imgCompleta[:,:,2]

    #Recorro imgCompleta y veo si esta dentro de los circulos en cada canal
    imgSegmendata = np.zeros((parteB.shape[0],parteB.shape[1], 3), dtype=np.uint8)

    for i in range(parteB.shape[0]):
        for j in range(parteB.shape[1]):
            if parteB[i,j] <= medB+rB and parteB[i,j] >= medB-rB:
                if parteG[i, j] <= medG + rG and parteG[i, j] >= medG - rG:
                    if parteR[i, j] <= medR + rR and parteR[i, j] >= medR - rR:
                        imgSegmendata[i,j,0] = parteB[i,j]
                        imgSegmendata[i,j,1] = parteG[i,j]
                        imgSegmendata[i,j,2] = parteR[i,j]

    graficar(imgCompleta[:, :, ::-1], 255, 0, 'gray', 'imagen original')
    graficar(imgSegmendata[:, :, ::-1], 255, 0, 'gray', 'imagen Segmentada')


# 4.2 Segmentación en HSV. En este caso se descarta la componente de luminosidad
# (I) y se segmenta en el plano HS.
# (a) Convierta la imagen al modelo HSV y visualice las componentes H y S.
# (b) Determine el subespacio rectangular a segmentar en el plano HS (utilice el
# histograma si lo considera pertinente).
# (c) Genere la máscara binaria recorriendo la imagen y verificando la pertenencia
# de cada pixel al rectángulo.
# (d) Obtenga la imagen segmentada mediante la aplicación de la máscara sobre
# la imagen original.
def ej4_2():
    # Extraigo la subimagen que quiero segmentar
    img = sub_image("../img/futbol.jpg")
    graficar(img[:, :, ::-1], 255, 0, 'gray', 'img')

    #La paso a HSV
    img = cv.cvtColor(img, cv.COLOR_BGR2HSV)

    #Centroides de H y S
    medH = np.mean(img[:,:,0])
    medS = np.mean(img[:,:,1])

    #Calculo los radios de  H y S como la varianza de cada componente
    rH = np.std(img[:,:,0])
    rS = np.std(img[:,:,1])

    # Abro la imagen completa
    imgCompleta = cv.imread("../img/futbol.jpg")
    imgCompleta = cv.cvtColor(imgCompleta, cv.COLOR_BGR2HSV)

    parteH = imgCompleta[:, :, 0]
    parteS = imgCompleta[:, :, 1]
    parteV = imgCompleta[:, :, 2]

    # Recorro imgCompleta y veo si esta dentro de los circulos en cada canal
    imgSegmendata = np.zeros((parteH.shape[0], parteS.shape[1], 3), dtype=np.uint8)

    for i in range(parteH.shape[0]):
        for j in range(parteS.shape[1]):
            if parteH[i,j] <= medH+rH and parteH[i,j] >= medH-rH:
                if parteS[i, j] <= medS + rS and parteS[i, j] >= medS - rS:
                    imgSegmendata[i,j,0] = parteH[i,j]
                    imgSegmendata[i,j,1] = parteS[i,j]
                    imgSegmendata[i, j, 2] = parteV[i,j]

    imgCompleta = cv.cvtColor(imgCompleta, cv.COLOR_HSV2BGR)
    graficar(imgCompleta[:, :, ::-1], 255, 0, 'gray', 'imagen original')

    imgSegmendata = cv.cvtColor(imgSegmendata, cv.COLOR_HSV2BGR)
    graficar(imgSegmendata[:, :, ::-1], 255, 0, 'gray', 'imagen Segmentada')


# 5 El gobierno de la provincia de Misiones lo ha contratado para realizar una aplicación
# que sea capaz de detectar zonas deforestadas. Para desarrollar un primer prototipo
# le han suministrado una imagen satelital (Deforestacion.png) en la que un experto
# ya delimitó el área donde deberı́a existir monte nativo y sobre la cual usted debe
# trabajar. Se requiere que su aplicación:
# • Segmente y resalte en algún tono de rojo el área deforestada.
# • Calcule el área total (hectáreas) de la zona delimitada, el área de la zona que
# tiene monte y el área de la zona deforestada.
# • (Opcional) Detecte automáticamente la delimitación de la zona.
# Ayuda:
# • Explore todos los canales de los diferentes modelos de color para determinar cual
# (o que combinación de ellos) le proporciona más información.
# • Como su objetivo es la segmentación de las distintas zonas, piense que her-
# ramienta (de las que ya conoce) le permitirı́a lograr zonas más homogéneas.
# • Utilice la referencia de la esquina inferior izquierda para computar los tamaños
# de las regiones.
def ej5():
    img = cv.imread("../img/Deforestacion.png", cv.IMREAD_GRAYSCALE)
    imgcolor = cv.imread("../img/Deforestacion.png")
    graficar(imgcolor[:,:,::-1], 255, 0, 'gray', 'imagen original')
    y = [152, 730]
    x = [275, 706]

    img[6:60,12:67] = 0
    img[74:317,24:54] = 0
    img[749:780,3:104] = 0
    graficar(img, 255, 0, 'gray', 'imagen original')

    # imgr = img[:,:,2]
    img[x[0]:x[1],y[0]:y[1]] = cv.equalizeHist(img[x[0]:x[1],y[0]:y[1]])
    graficar(img, 255, 0, 'gray', 'imagen en escala de grises ecualizada')

    blur = cv.GaussianBlur(img, (33,33), 0)
    graficar(blur, 255, 0, 'gray', 'imagen ecualizada mas desenfoque')

    ret, umbral = cv.threshold(blur, 190, 255, cv.THRESH_BINARY)

    graficar(umbral, 255, 0, 'gray', 'imagen desenfocada umbralizada')

    area = cv.countNonZero(umbral)

    porcDeforestado = 1.0*area/(umbral[x[0]:x[1],y[0]:y[1]].shape[0]*umbral[x[0]:x[1],y[0]:y[1]].shape[1])
    print "Area deforestada: ",100.0*porcDeforestado,"%"
    areaPixel = pow(200 / 90.0,2)
    hectareas = areaPixel * area * 0.0001
    print "Hectareas: ", hectareas

    porc_transparencia = .85

    imgcolor[x[0]:x[1],y[0]:y[1],2] = porc_transparencia * umbral[x[0]:x[1],y[0]:y[1]] + (1-porc_transparencia) * imgcolor[x[0]:x[1],y[0]:y[1],2]

    graficar(imgcolor[:,:,::-1], 255, 0, 'gray', 'Area deforestada en rojo')


if __name__ == '__main__':
    ej5()
    plt.show()
