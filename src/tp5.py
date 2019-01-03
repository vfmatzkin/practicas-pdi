# -*- coding: utf-8 -*-

import pdifun as fun
import cv2 as cv
from matplotlib import pyplot as plt
import funciones as func
import numpy as np
import math


# Ej 1.1: Construya imágenes binarias de: una linea horizontal, una linea vertical, un
# cuadrado centrado, un rectángulo centrado, un circulo.
# 1.2 A partir de las imágenes anteriores, calcule y visualice la TDF, haciendo
# hipótesis sobre la imagen a obtener antes de la visualización. En todos los
# casos varı́e las dimensiones y localización de los objetos, y repita el análisis.
def ej1_1y1_2():
    linea_vertical = np.zeros([512,512],dtype=np.uint8)
    linea_vertical[0:512,256] = 255

    linea_horizontal = np.zeros([512,512],dtype=np.uint8)
    linea_horizontal[256, 0:512] = 255

    circulo = np.zeros([512,512],dtype=np.uint8)
    cv.circle(circulo, (256,256), 50, 255, -1)
    cv.circle(circulo, (256,256), 49, 0, -1)

    rectangulo = np.zeros([512,512],dtype=np.uint8)
    cv.rectangle(rectangulo, (156,206), (356, 306), 255,3)

    rectanguloC = np.zeros([512, 512], dtype=np.uint8)
    cv.rectangle(rectanguloC, (100, 206), (300, 306), 255, 3)

    cuadrado = np.zeros([512,512],dtype=np.uint8)
    cv.rectangle(cuadrado, (206,206), (306, 306), 255,3)

    lv_fft = fun.spectrum(linea_vertical)
    lh_fft = fun.spectrum(linea_horizontal)
    cr_fft = fun.spectrum(circulo)
    rc_fft = fun.spectrum(rectangulo)
    rc_fft2 = fun.spectrum(rectanguloC)
    cd_fft = fun.spectrum(cuadrado)

    func.graficar(linea_vertical, 255, 0, 'gray', 'Linea Vertical')
    func.graficar(lv_fft, lv_fft.max(), 0, 'gray', 'Linea Vertical Transformada')

    func.graficar(linea_horizontal, 255, 0, 'gray', 'Linea Horizontal')
    func.graficar(lh_fft, lh_fft.max(), 0, 'gray', 'Linea Horizontal Transformada')

    func.graficar(circulo, 255, 0, 'gray', 'Circulo')
    func.graficar(cr_fft, cr_fft.max(), 0, 'gray', 'Circulo Transformada')

    func.graficar(rectangulo, 255, 0, 'gray', 'Rectangulo')
    func.graficar(rc_fft, rc_fft.max(), 0, 'gray', 'Rectangulo Transformada')

    func.graficar(rectanguloC, 255, 0, 'gray', 'Rectangulo corrido')
    func.graficar(rc_fft2, rc_fft2.max(), 0, 'gray', 'Rectangulo Transformada Corrida')

    func.graficar(cuadrado, 255, 0, 'gray', 'Cuadrado')
    func.graficar(cd_fft, cd_fft.max(), 0, 'gray', 'Cuadrado Transformada')


# 1.3 Cree una imagen de 512x512 conteniendo una linea vertical blanca centrada
# de un pixel de ancho sobre un fondo negro. Rote la imagen 20 grados y
# extraiga una sección de 256x256 de la imagen original y de la imagen rotada,
# de manera que las lineas tengan sus extremos en los bordes superior e inferior,
# sin margenes. Visualice la TDF de ambas imágenes. Explique, utilizando
# argumentos intuitivos, por qué las magnitudes de Fourier aparecen como lo
# hacen en las img, y a qué se deben las diferencias.
def ej1_3():
    cols = 512
    rows = 512
    linea_horizontal = np.zeros([512, 512], dtype=np.uint8)
    linea_horizontal[256, 0:512] = 255

    M = cv.getRotationMatrix2D((cols / 2, rows / 2), 20, 1)
    linea_horizontal_r = cv.warpAffine(linea_horizontal, M, (cols, rows))

    lh_rec = linea_horizontal[128:382,128:382]
    lh_rec_r = linea_horizontal_r[128:382,128:382]
    func.graficar(lh_rec, 255, 0, 'gray', 'Linea Horizontal Recortada')
    func.graficar(lh_rec_r, 255, 0, 'gray', 'Linea Horizontal Recortada Rotada')


    lh_fft = fun.spectrum(lh_rec)
    lh_r_fft = fun.spectrum(lh_rec_r)
    func.graficar(lh_fft, lh_fft.max(), 0, 'gray', 'Linea Horizontal Transformada')
    func.graficar(lh_r_fft, lh_r_fft.max(), 0, 'gray', 'Linea Horizontal Transformada')


# 1.4 Cargue diferentes imágenes y visualice la magnitud de la TDF. Infiera, a
# grandes rasgos, la correspondencia entre componentes frecuenciales y detalles
# de las imágenes.
def ej1_4():
    img1 = cv.imread("../img/imagenB.tif", cv.IMREAD_GRAYSCALE)
    img2 = cv.imread("../img/cameraman.tif", cv.IMREAD_GRAYSCALE)
    img3 = cv.imread("../img/hubble.tif", cv.IMREAD_GRAYSCALE)
    img4 = cv.imread("../img/chairs.jpg", cv.IMREAD_GRAYSCALE)

    sp1 = fun.spectrum(img1)
    sp2 = fun.spectrum(img2)
    sp3 = fun.spectrum(img3)
    sp4 = fun.spectrum(img4)

    func.graficar(img1, img1.max(), 0, 'gray', 'Original: imagenB.tif')
    func.graficar(sp1, sp1.max(), 0, 'gray', 'FFT: imagenB.tif')
    func.graficar(img2, img2.max(), 0, 'gray', 'Original: cameraman.tif')
    func.graficar(sp2, sp2.max(), 0, 'gray', 'FFT: cameraman.tif')
    func.graficar(img3, img3.max(), 0, 'gray', 'Original: hubble.tif')
    func.graficar(sp3, sp3.max(), 0, 'gray', 'FFT: hubble.tif')
    func.graficar(img4, img4.max(), 0, 'gray', 'Original: chairs.jpg')
    func.graficar(sp4, sp4.max(), 0, 'gray', 'FFT: chairs.jpg')


# 2.1: Construya un filtro pasa-bajos ideal mediante la especificación de un circulo
# de altura 1 sobre una matriz de ceros. Cargue la imagen y filtrela en el
# dominio de la frecuencia, obteniendo mediante la TDF inversa una versión
# suavizada de la imagen. Muestre en un mismo gráfico la imagen original con
# la filtrada y compare. Repita el ejercicio para diversas frecuencias de corte.
# Compruebe la aparición del fenómeno de Gibbs.
def ej2_1():
    img = cv.imread("../img/huang1.jpg", cv.IMREAD_GRAYSCALE)

    rows, cols = img.shape

    # Filtros pasa altos: Para que funcionen, comentar los pasa bajos.
    # imgf1 = fun.filterImg(img, 1-fun.filterIdeal(rows,cols,0.08))
    # spec1 = fun.spectrum(fun.filterImg(img,fun.filterIdeal(rows,cols,0.15)))
    #
    # imgf2 = fun.filterImg(img, 1-fun.filterIdeal(rows,cols,0.15))
    # spec2 = fun.spectrum(fun.filterImg(img, fun.filterIdeal(rows,cols,0.15)))
    #
    # imgf3 = fun.filterImg(img, 1-fun.filterIdeal(rows,cols,0.6))
    # spec3 = fun.spectrum(fun.filterImg(img, fun.filterIdeal(rows,cols,0.6)))


    # Filtros pasa bajos
    imgf1 = fun.filterImg(img, fun.filterIdeal(rows,cols,0.08))
    spec1 = fun.spectrum(fun.filterImg(img,fun.filterIdeal(rows,cols,0.15)))

    imgf2 = fun.filterImg(img, fun.filterIdeal(rows,cols,0.15))
    spec2 = fun.spectrum(fun.filterImg(img, fun.filterIdeal(rows,cols,0.15)))

    imgf3 = fun.filterImg(img, fun.filterIdeal(rows,cols,0.6))
    spec3 = fun.spectrum(fun.filterImg(img, fun.filterIdeal(rows,cols,0.6)))


    func.graficar(imgf1, imgf1.max(), 0, 'gray', 'Imagen para 0.1')
    func.graficar(spec1, (spec1).max(), 0, 'gray', 'Espectro para 0.08')

    func.graficar(imgf2, imgf2.max(), 0, 'gray', 'Imagen para 0.15')
    func.graficar(spec2, (spec2).max(), 0, 'gray', 'Espectro para 0.15')

    func.graficar(imgf3, imgf3.max(), 0, 'gray', 'Imagen para 0.6')
    func.graficar(spec3, (spec3).max(), 0, 'gray', 'Espectro para 6')


# 2.2 Construya un filtro pasa-bajos tipo Butterworth utilizando la definición en
# frecuencia. Filtre la imagen, modificando la frecuencia de corte y compro-
# bando el efecto sobre la imagen filtrada. Verifique el efecto del filtro respecto
# al fenómeno de Gibbs.
def ej2_2():
    img = cv.imread("../img/huang1.jpg", cv.IMREAD_GRAYSCALE)

    #Cantidad de filas y columnas de la imagen
    rows, cols = img.shape

    #Variables del filtro
    corte = 0.02
    order = 5

    #Creo el filtro Butterworth pasa bajos en frecuencia
    filtro = fun.filterButterworth(rows, cols, corte, order)

    # Lo convierto a pasa altos
    # filtro = 1 - filtro

    func.graficar(filtro, filtro.max(), 0, 'gray', 'Filtro')

    #Filtro la imagen en freciencia
    imgFiltrada = fun.filterImg(img, filtro)

    #Calculo el espectro de la imagen filtrada
    espectro = fun.spectrum(imgFiltrada)

    #Grafico
    func.graficar(img, img.max(), 0, 'gray', 'Imagen Original')
    func.graficar(imgFiltrada, imgFiltrada.max(), 0, 'gray', 'Imagen filtrada Corte: 0.02. Orden: 5.')
    func.graficar(espectro, espectro.max(), 0, 'gray', 'Espectro')


# 2.3 A partir de la función de transferencia h(x,y) de un filtro gaussiano pasa-
# bajos obtenga la respuesta en frecuencia aplicando la TDF, con tamaño igual
# al de la imagen. Calcule y visualice la TDF de la imagen y el producto de
# transformadas, verificando la acción del filtro. Obtenga la imagen filtrada y
# compare con la imagen original.
def ej2_3():
    img = cv.imread("../img/huang1.jpg", cv.IMREAD_GRAYSCALE)

    #Cantidad de filas y columnas de la imagen
    rows, cols = img.shape

    #Variables del filtro
    corte = 0.01

    #Creo el filtro Butterworth en frecuencia
    filtro = fun.filterGaussian(rows, cols, corte)

    # Pasa altos
    # filtro = 1 - filtro

    func.graficar(filtro, filtro.max(), 0, 'gray', 'Filtro')

    #Filtro la imagen en freciencia
    imgFiltrada = fun.filterImg(img, filtro)

    #Calculo el espectro de la imagen filtrada
    espectro = fun.spectrum(imgFiltrada)

    #Grafico
    func.graficar(img, img.max(), 0, 'gray', 'Imagen Original')
    func.graficar(imgFiltrada, imgFiltrada.max(), 0, 'gray', 'Imagen filtrada Corte: 0.02')
    func.graficar(espectro, espectro.max(), 0, 'gray', 'Espectro')


# 2.4  Repita el ejercicio anterior para filtro gaussiano definido en frecuencia.
def ej2_4():
    img = cv.imread("../img/huang1.jpg", cv.IMREAD_GRAYSCALE)

    # Cantidad de filas y columnas de la imagen
    rows, cols = img.shape

    # Variables del filtro
    corte = 0.010

    # Creo el filtro Gauss en frecuencia
    filtro = fun.filterGaussian(rows, cols, corte)

    # Filtro pasa altos
    # filtro = 1- filtro

    func.graficar(filtro, filtro.max(), 0, 'gray', 'Filtro')

    # Filtro la imagen en freciencia
    imgFiltrada = fun.filterImg(img, filtro)

    # Calculo el espectro de la imagen filtrada
    espectro = fun.spectrum(imgFiltrada)

    # Grafico
    func.graficar(img, img.max(), 0, 'gray', 'Imagen Original')
    func.graficar(imgFiltrada, imgFiltrada.max(), 0, 'gray', 'Imagen filtrada')
    func.graficar(espectro, espectro.max(), 0, 'gray', 'Espectro')


# 2.5 Repita los puntos 1. a 4. para filtros pasa-altos.
# Solución: Descomentar en los puntos anteriores la linea posterior a la creación
# del filtro para convertirlo en pasa altos.


# 3.1 A partir de la definición de una máscara de filtrado pasa-altos en el dominio
# espacial, obtenga la función de transferencia correspondiente a un filtro de
# alta potencia según Hap = (A − 1) + Hpa , y a un filtro de énfasis de alta
# frecuencia según: Heaf = a + b Hpa
def ej3_1():
    img = cv.imread("../img/huang1.jpg", cv.IMREAD_GRAYSCALE)

    # Cantidad de filas y columnas de la imagen
    rows, cols = img.shape

    # Variables del filtro
    corte = .05

    # Creo el filtro Gauss en frecuencia
    filtro = fun.filterGaussian(rows, cols, corte)
    filtroPasaAlto = 1 - filtro

    a = 1.1
    A = a * np.ones([rows, cols])
    Hap = (A - 1) + filtroPasaAlto

    # Filtro la imagen en freciencia
    imgFiltrada = fun.filterImg(img, Hap)

    # Calculo el espectro de la imagen filtrada
    espectro = fun.spectrum(imgFiltrada)

    # Grafico
    func.graficar(img, img.max(), 0, 'gray', 'Imagen Original')
    func.graficar(filtroPasaAlto, filtroPasaAlto.max(), 0, 'gray', 'Filtro pasa alto')
    func.graficar(imgFiltrada, imgFiltrada.max(), 0, 'gray', 'Imagen filtrada (filtro de alta frecuencia)')
    func.graficar(espectro, espectro.max(), 0, 'gray', 'Espectro')


# 3.2  Elija apropiadamente los valores de los parámetros A, a y b y aplique los
# filtros a la imagen ’camaleon.tif’, visualizando la imagen original junto a
# su TDF, y la imagen resultante con su TDF.
def ej3_2(show_fft = True):
    img = cv.imread("../img/camaleon.tif", cv.IMREAD_GRAYSCALE)

    # Cantidad de filas y columnas de la imagen
    rows, cols = img.shape

    # Variables del filtro
    corte = 0.05

    # Creo el filtro Gauss en frecuencia
    filtro = fun.filterGaussian(rows, cols, corte)
    filtroPasaAlto = np.ones([rows, cols]) - filtro

    #Creo filtro de alta potencia
    a = 0.5
    b = 10

    filtroEAF = a + b * filtroPasaAlto

    # Filtro la imagen en freciencia
    imgFiltrada = fun.filterImg(img, filtroEAF)

    #TDF de la imagen original
    espectro = fun.spectrum(img)

    # TDF de la imagen filtrada
    espectroFiltrada = fun.spectrum(imgFiltrada)


    func.graficar(img, 255, 0, 'gray', 'Imagen Original')
    func.graficar(imgFiltrada, 255, 0, 'gray', 'Imagen Filtrada (Alta potencia frecuencial)')

    if show_fft:
        func.graficar(espectro, espectro.max(), 0, 'gray', 'TDF imagen original')
        func.graficar(espectroFiltrada, espectroFiltrada.max(), 0, 'gray', 'TDF Imagen Filtrada')


# 3.3  Compare la imagen de alta potencia con la que se obtiene al aplicar el filtro
# en el dominio espacial.
def ej3_3():
    img = cv.imread("../img/camaleon.tif", cv.IMREAD_GRAYSCALE)
    # Defino el filtro en el espacio
    n = 5
    filtered = cv.filter2D(img, -1, np.ones((n, n), np.float32) / (n * n))

    # Coeficiente del filtro de alta potencia
    A = 2.0

    # Filtro de alta potencia
    img2 = cv.subtract(A * img, filtered.astype(np.float))

    func.graficar(img2, 255, 0, 'gray', 'Imagen filtrada (Alta potencia espacial)')

    # Filtro de alta potencia en frecuencia
    ej3_2(show_fft = False)

# 4 Filtrado homomórfico
# Un uso extendido del filtrado homomórfico es la corrección de iluminación no
# uniforme en distintas zonas de la imagen, generalmente con alto contenido de
# información en la zona de bajo brillo, como por ej. en filmaciones de cámaras de
# seguridad y fotos con luz de día con sol de frente. En imágenes de este tipo, el
# filtro homomórfico corrige el contraste en la zona de interés y acentúa los detalles
# simultáneamente.
# 1. Genere la función de transferencia H que caracteriza a un filtro homomórfico.
# 2. Aplique el proceso en las imágenes ‘casilla.tif’ y ‘reunion.tif’, con
# valores apropiados de gL , gH , D0 y orden (prueba y error en cada imagen...).
# 3. Verifique las bondades del método comparando el resultado anterior con la
# imagen que se obtiene al ecualizar la imagen original. Esta técnica suele ser
# eficaz con determinadas imágenes si el resultado se procesa con alguna técnica
# de manipulación de histogramas, fundamentalmente expansión o igualación.
def ej4():
    #Filtrado Homomorfico
    img_o = cv.imread("../img/reunion.tif", cv.IMREAD_GRAYSCALE)
    img = np.array(img_o, dtype=np.float32)
    rows, cols = img.shape

    # Logaritmo de la imagen original
    img_log = np.log(1+img)

    # Variables del filtro
    corte = 0.01
    order = 5
    gL = 0.5
    gH = 0.6

    # Filtro que voy a usar en frecuencia (basado en el gaussiano)
    filtro = func.filterHomomorfico(rows, cols, corte, gL, gH, order)

    # Filtro la imagen
    imgFiltrada = fun.filterImg(img_log, filtro)

    # Saco el logaritmo
    imgFinal = np.exp(imgFiltrada - imgFiltrada.min())
    imgFinal = 255 * imgFinal/imgFinal.max()

    # La imagen se abrió como flotante para poder manipular sus números con mayor precisión
    # Luego de operar, se debe volver a 8 bits entero sin signo .
    img = np.array(img, dtype=np.uint8)
    imgFinal = np.array(imgFinal, dtype=np.uint8)

    # Ecualizacion (Ver si mejora)
    imgeq = cv.equalizeHist(imgFinal)

    # Graficar
    func.graficar(img, 255, 0, 'gray', 'Imagen original')
    func.graficar(imgFinal, 255, 0, 'gray', 'Imagen Final')
    func.graficar(imgeq, 255, 0, 'gray', 'Imagen Final ecualizada')


if __name__ == '__main__':
    ej4()
    plt.show()
