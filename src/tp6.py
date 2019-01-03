# -*- coding: utf-8 -*-


import cv2 as cv
from matplotlib import pyplot as plt
import funciones as func
import numpy as np
from click_and_crop import *
import math
import pdifun as fun

# Modelos de ruido
# 1.1 Genere imágenes con diferentes tipos de ruido, y estudie la distribución obtenida
# analizando el histograma

def ej1_1():
    # Implementación de distintos tipos de ruido: Descomentar la gráfica del que sea de interés

    # Cargar imagen
    img = cv.imread("../img/cameraman.tif", cv.IMREAD_GRAYSCALE)

    # Ruidos (copiar imagen y generar ruido)
    salYpimienta = img.copy()
    salYpimienta = func.ruidoSalPimienta(salYpimienta, 0.5, 0.03)

    gauss = img.copy()
    gauss = func.ruidoGaussiano(gauss, 0, 0.02)

    raileigh = img.copy()
    raileigh = func.ruidoRayleigh(raileigh, 0.5)

    uniforme = img.copy()
    uniforme = func.ruidoUniforme(uniforme, 0, 0.5)

    exponencial = img.copy()
    exponencial = func.ruidoExponencial(exponencial, 0.09)

    gamma = img.copy()
    gamma = func.ruidoGamma(gamma, 1, 0.2)

    # Graficar
    func.graficar(img, 255, 0, 'gray', 'Imagen de entrada')
    func.histograma(img, 'Imagen de entrada')

    # func.graficar(salYpimienta, 255, 0, 'gray', 'Ruido Sal y pimienta')
    # func.histograma(salYpimienta, 'Ruido sal y pimienta')

    # func.graficar(gauss, 255, 0, 'gray', 'Ruido Gaussiano')
    # func.histograma(gauss, 'Ruido gaussiano')

    # func.graficar(raileigh, 255, 0, 'gray', 'Ruido Raileigh')
    # func.histograma(raileigh, 'Ruido Raileigh')

    # func.graficar(uniforme, 255, 0, 'gray', 'Ruido Uniforme')
    # func.histograma(uniforme, 'Ruido Uniforme')

    # func.graficar(exponencial, 255, 0, 'gray', 'Ruido Exponencial')
    # func.histograma(exponencial, 'Ruido Exponencial')

    func.graficar(gamma, 255, 0, 'gray', 'Ruido Gamma')
    func.histograma(gamma, 'Ruido Gamma')


# 1.2 . Genere un patron de grises constantes y sume el ruido generado previamente.
# Varie los parametros del ruido (media, desvio, etc.) y verifique los efectos en el
# histograma de porciones de grises constantes.

# Respuesta: Está en las diapositivas (análisis de las zonas homogéneas de una imagen
# con cada uno de los ruidos).

# 2.1 Implemente los filtros de la media geométrica y de la media contra-armónica.
# 2.2  Genere una imagen ruidosa a partir de ‘sangre.jpg’, contaminándola con mez-
# cla de ruido impulsivo y gaussiano.
def ej2_1_y_2():
    img = cv.imread("../img/sangre.jpg", cv.IMREAD_GRAYSCALE).astype(np.float16)

    # Se ensucia con ruido Gaussiano
    img_gau = func.ruidoGaussiano(img.copy(), 0, 0.05)
    img_limpia_g = func.filtroMediaGeometrica(img_gau.copy(), 3, 3)

    # Se ensucia con ruido Impulsivo
    img_sp = func.ruidoSalPimienta(img.copy(), 1, 0.5)
    img_limpia_s = func.filtroMediaContraarmonica(img_sp.copy(), -1, 3, 3)

    # Para graficar vuelvo a enteros sin signo
    img = img.astype(np.uint8)
    img_gau = img_gau.astype(np.uint8)
    img_limpia_g = img_limpia_g.astype(np.uint8)
    img_sp = img_sp.astype(np.uint8)
    img_limpia_s = img_limpia_s.astype(np.uint8)

    func.graficar(img, 255, 0, 'gray', 'Imagen de entrada')
    func.graficar(img_gau, 255, 0, 'gray', 'Imagen con ruido Gaussiano')
    func.graficar(img_sp, 255, 0, 'gray', 'Imagen con ruido Sal')
    func.graficar(img_limpia_g, 255, 0, 'gray', 'Imagen con ruido Gaussiano limpia')
    func.graficar(img_limpia_s, 255, 0, 'gray', 'Imagen con ruido Sal limpia')

# 2.3  Aplique los filtros y verifique la restauracion mediante la comparacion del ECM
# entre la imagen filtrada y la limpia vs. el ECM entre la imagen degradada y la
# limpia.

# Solución: Hay una funcion para el ECM pero no se si esta bien porque la probe para el
# ejercicio 2.2 y daba mas alto con la imagen limpia que con la sucia.


# 3 Filtros de orden
# 1. Los siguientes casos listan secuencias de procesamientos cuyos efectos sobre una
# imagen degradada son, en principio, similares:
# (a) Filtro de mediana y filtro del punto medio.
# (b) Filtro de la media-alfa recortado.
# 2. implemente los filtros mencionados y apliquelos a la misma imagen degradada
# del ejercicio anterior.
# 3. indique en cual de los casos se logra una mayor remocion del ruido.
# 4. Compare con el resultado del filtrado de medias del Ejercicio 2.

# Solucion: No lo hicimos porque ya aplicamos muchas veces el de mediana


# 4 Eliminación de ruido periódico
# 4.1. La imagen ‘img degradada.tif’ está altamente degradada por interferencia
# sinusoidal. Muestre la imagen junto a su espectro de Fourier y analice la infor-
# mación del ruido.
# 4.2. Localice los picos fundamentales del ruido.
# 4.3. Implemente un filtro rechazabanda o un filtro notch que elimine las zonas cen-
# trales de frecuencias del ruido (y su conjugado).
def ej4_3_notch():
    img = cv.imread("../img/img_degradada.tif", cv.IMREAD_GRAYSCALE)
    rows, cols = img.shape

    # Espectro de la imagen
    # Shiftear en false para poder determinar el punto del espectro a anular
    img_fft = fun.spectrum(img, shiftear=False)

    # Elegir puntos a filtrar con clic
    punto1 = func.elegir_punto(img_fft)

    fNotch1 = fun.filtroNotch(rows, cols, punto1)

    img_fil = fun.filterImg(img,fNotch1)

    # Calcular el espectro para graficarlo
    img_fil_s = fun.spectrum(img_fil)

    # Graficar
    func.graficar(img, 255, 0, 'gray', 'Imagen original')
    func.graficar(img_fft, 1, 0, 'gray', 'Espectro de la imagen original')
    # func.graficar(fNotch1, 1, 0, 'gray', 'Filtro notch')
    func.graficar(img_fil, 255, 0, 'gray', 'Imagen filtrada')
    func.graficar(img_fil_s, 1, 0, 'gray', 'Espectro de la imagen filtrada')


def ej4_3_rechazabanda():
    img = cv.imread("../img/img_degradada.tif", cv.IMREAD_GRAYSCALE)
    rows, cols = img.shape

    # Espectro de la imagen
    # Shiftear en false para poder determinar el punto del espectro a anular
    img_fft = fun.spectrum(img, shiftear=False)

    # Parametros de los filtros Butterworth (dentro del pasabanda)
    cortes = [.04, .043]
    orden = 100

    # Obtengo el filtro y filtro la imagen
    filtro = fun.filtroRechazaBanda(rows, cols, cortes, orden)
    img_fil = fun.filterImg(img, filtro)

    # Calcular el espectro para graficarlo
    img_fil_s = fun.spectrum(img_fil, shiftear=False)

    # Graficar
    func.graficar(img, 255, 0, 'gray', 'Imagen original')
    func.graficar(img_fft, 1, 0, 'gray', 'Espectro de la imagen original')
    func.graficar(filtro, 1, 0, 'gray', 'Filtro notch')
    func.graficar(img_fil, 255, 0, 'gray', 'Imagen filtrada')
    func.graficar(img_fil_s, 1, 0, 'gray', 'Espectro de la imagen filtrada')


# 4.4. Una solución alternativa (algunas veces más efectiva, pero menos formal) con-
# siste en construir un filtro notch ad-hoc para el espectro bajo análisis. El método
# consiste simplemente en hacer cero las frecuencias del ruido (sean zonas circu-
# larmente simétricas o no) .
# 4.5. Compare cualitativamente la imagen de salida obtenida con la imagen original
# ‘img.tif’ y cuantitativamente mediante el cálculo del error cuadrático medio.
# 4.6. Obtenga la imagen de sólo ruido mediante la aplicación de un filtro pasabanda
# o un filtro notch pasante.
def ej4_4():
    img = cv.imread("../img/img_degradada.tif", cv.IMREAD_GRAYSCALE)
    rows, cols = img.shape

    # Espectro de la imagen
    # Shiftear en false para poder determinar el punto del espectro a anular
    img_fft = fun.spectrum(img, shiftear=False)

    # Elegir puntos a filtrar con clic
    punto1 = func.elegir_punto(img_fft)

    fNotch1 = fun.filtroNotch(rows, cols, punto1, pixel = True)

    img_fil = fun.filterImg(img,fNotch1)

    # Calcular el espectro para graficarlo
    img_fil_s = fun.spectrum(img_fil, shiftear = False)

    # Graficar
    func.graficar(img, 255, 0, 'gray', 'Imagen original')
    func.graficar(img_fft, 1, 0, 'gray', 'Espectro de la imagen original')
    func.graficar(img_fil, 255, 0, 'gray', 'Imagen filtrada')
    func.graficar(img_fil_s, 1, 0, 'gray', 'Espectro de la imagen filtrada')


# 4.7. Repita el ejercicio para las imágenes ‘noisy moon’ y ‘HeadCT degradada’.
# Respuesta: Noisy moon es una cagada porque hay que filtrar una franja vertical y
# otra horizontal fijas, así que no lo hicimos.
def ej4_7():
    img = cv.imread("../img/HeadCT_degradada.tif", cv.IMREAD_GRAYSCALE)
    img_fft = fun.spectrum(img)
    img_fft = np.fft.ifftshift(img_fft)

    # Cantidad de filas y columnas de la imagen
    rows, cols = img.shape

    # En este caso el filtro se construye a mano, y se anulan los puntos blancos
    # del espectro que no tienen nada que ver con su entorno
    filtro = np.ones((rows, cols))
    filtro[40,40] = 0
    filtro[20,0] = 0
    filtro[0,10] = 0
    filtro[492,0] = 0

    mascara = filtro * img_fft

    # Aplico el filtro
    filtrada = fun.filterImg(img,filtro)

    # Graficar
    func.graficar(img, 255, 0, 'gray', 'Imagen original')
    func.graficar(img_fft, 1, 0, 'gray', 'Espectro')
    func.graficar(filtrada, 255, 0, 'gray', 'filtrada')
    func.graficar(mascara, 1, 0, 'gray', 'mascara')


# 5: Restauracion de desenfoque por movimiento
# Se tiene la imagen ‘huang3 movida.tif’ que ha sido degradada por desenfoque
# debido a movimiento lineal uniforme. Restaure la imagen mediante filtrado pseudo-
# inverso, asumiendo que la funcion de degradacion se puede estimar por modelado
# como se muestra en el pdf.

# Ejercicio incompleto: Falta lo más importante
def ej5():
    img = cv.imread("../img/huang3_movida.tif", cv.IMREAD_GRAYSCALE)
    rows, cols = img.shape
    img_fft = fun.spectrum(img)
    img_fft = np.fft.ifftshift(img_fft)

    func.graficar(img, 255, 0, 'gray', 'imagen original')
    func.graficar(img_fft, 1, 0, 'gray', 'Espectro')


# 6. Para las imagenes FAMiLiA a.jpg, FAMiLiA b.jpg y FAMiLiA c.jpg, identifique el
# tipo de ruido que afecta a cada una y calcule los parametros estadisticos para dichos
# ruidos. Restaure las imagenes.
def ej6_A():
    imgA = cv.imread("../img/FAMILIA_a.jpg", cv.IMREAD_GRAYSCALE)
    sub_imgA = sub_image("../img/FAMILIA_a.jpg")

    # Analizo el histograma de cada sub imagen
    func.histograma(sub_imgA, "Histograma A")

    #Calculo los parametros estadisticos para cada sub imagen
    mediaA = np.mean(sub_imgA)
    varianzaA = np.var(sub_imgA)
    print 'Media y varianza original'
    print mediaA, varianzaA

    #La imagen A tiene solo ruido gausiano
    # Probando con un filtro de 5x5
    sub_imgA5x5 = cv.GaussianBlur(sub_imgA,(5,5),0)
    imgAGauss5x5 = cv.GaussianBlur(imgA,(5,5),0)

    mediaGauss5x5 = np.mean(sub_imgA5x5)
    varianzaGauss5x5 = np.var(sub_imgA5x5)
    print 'Media y varianza Gauss5x5'
    print mediaGauss5x5, varianzaGauss5x5

    func.histograma(sub_imgA5x5, "A Despues de filtro gausiano 5x5")

    # Probando con un filtro de 9x9
    sub_imgA9x9 = cv.GaussianBlur(sub_imgA, (9, 9), 0)
    imgAGauss9x9 = cv.GaussianBlur(imgA, (9, 9), 0)

    mediaGauss9x9 = np.mean(sub_imgA9x9)
    varianzaGauss9x9 = np.var(sub_imgA9x9)
    print 'Media y varianza Gauss9x9'
    print mediaGauss9x9, varianzaGauss9x9

    func.histograma(sub_imgA9x9, "A Despues de filtro gausiano 9x9")

    # Aplico filtro mediana a los a los 2 resultados de filtrado gausiano
    # Al de 5x5
    sub_imgA5x5median = cv.medianBlur(sub_imgA5x5, 5)
    imgA5x5median = cv.medianBlur(imgAGauss5x5, 5)

    mediaGauss5x5median = np.mean(sub_imgA5x5median)
    varianzaGauss5x5median = np.var(sub_imgA5x5median)
    print 'Media y varianza Gauss5x5 y mediana'
    print mediaGauss5x5median, varianzaGauss5x5median

    func.histograma(sub_imgA5x5median, "A Despues de gauss 5x5 y mediana")

    # Al de 9x9
    sub_imgA9x9median = cv.medianBlur(sub_imgA9x9, 5)
    imgA9x9median = cv.medianBlur(imgAGauss9x9, 5)

    mediaGauss9x9median = np.mean(sub_imgA9x9median)
    varianzaGauss9x9median = np.var(sub_imgA9x9median)
    print 'Media y varianza Gauss9x9 y mediana'
    print mediaGauss9x9median, varianzaGauss9x9median

    func.histograma(sub_imgA9x9median, "A Despues de gauss 9x9 y mediana")

    # Aplico filtro punto medio despues de Gausiano
    sub_imgA = cv.cvtColor(sub_imgA, cv.COLOR_BGR2GRAY)

    sub_imgAPM = func.filter_midPoint3(sub_imgA)
    imgAPM = func.filter_midPoint3(imgA)

    mediaPM = np.mean(sub_imgAPM)
    varianzaPM = np.var(sub_imgAPM)
    print 'Media y varianza PM'
    print mediaPM, varianzaPM

    func.histograma(sub_imgAPM, "A Despues punto medio")

    #sub_imgA9x9 = cv.cvtColor(sub_imgA9x9, cv.COLOR_BGR2GRAY)

    #sub_imgA9x9PM = func.filter_midPoint3(sub_imgA9x9)
    #imgA9x9PM = func.filter_midPoint3(imgAGauss9x9)

    #mediaGauss9x9PM = np.mean(sub_imgA9x9PM)
    #varianzaGauss9x9PM = np.var(sub_imgA9x9PM)
    #print 'Media y varianza Gauss9x9 y PM'
    #print mediaGauss9x9PM, varianzaGauss9x9PM

    #plt.figure()
    #plt.hist(sub_imgA9x9PM.flatten(), 255)
    #plt.title('A Despues de gauss 9x9 y punto medio')
    #plt.show()

    #Grafico
    func.graficar(imgA, 255, 0, 'gray', 'Original A')
    func.graficar(sub_imgA, 255, 0, 'gray', 'Zona homogenea de analisis')
    func.graficar(imgAGauss5x5, 255, 0, 'gray', 'A con filtro gaussiano 5x5')
    func.graficar(imgAGauss9x9, 255, 0, 'gray', 'A con filtro gaussiano 9x9')
    func.graficar(imgA5x5median, 255, 0, 'gray', 'A con filtro gaussiano 5x5 y mediana')
    func.graficar(imgAPM, 255, 0, 'gray', 'A con filtro Punt Medio')
    func.graficar(imgA9x9median, 255, 0, 'gray', 'A con filtro gaussiano 9x9 y mediana')
    #func.graficar(imgA9x9PM, 255, 0, 'gray', 'A con filtro gaussiano 9x9 y Punt Medio')


def ej6_B():
    # Analisis de B
    imgB = cv.imread("../img/FAMILIA_b.jpg", cv.IMREAD_GRAYSCALE)
    func.graficar(imgB, 255, 0, 'gray', 'Original B')

    # Extraigo la subimagen que quiero segmentar de cada imagen a analizar
    sub_imgB = sub_image("../img/FAMILIA_b.jpg")
    sub_imgB = cv.cvtColor(sub_imgB, cv.COLOR_BGR2GRAY)

    # Analizo el histograma de sub imagen
    func.histograma(sub_imgB, "Histograma B")

    #Calculo los parametros estadisticos para cada sub imagen
    mediaB = np.mean(sub_imgB)
    varianzaB = np.var(sub_imgB)
    print mediaB, varianzaB

    #La imagen B tiene ruido uniforme y gausiano
    sub_imgB_midPoint = func.filter_midPoint3(sub_imgB)
    imgB_midPoint = func.filter_midPoint3(imgB)

    mediaB_midPoint = np.mean(sub_imgB_midPoint)
    varianzaB_midPoint = np.var(sub_imgB_midPoint)
    print mediaB_midPoint, varianzaB_midPoint

    func.histograma(sub_imgB_midPoint, "B Despues de filtro de punto medio")

    sub_imgB_adaptative = func.filter_adaptative(sub_imgB, varianzaB)
    imgB_adaptative = func.filter_adaptative(imgB, varianzaB)

    mediaB_adaptative = np.mean(sub_imgB_adaptative)
    varianzaB_adaptative = np.var(sub_imgB_adaptative)
    print mediaB_adaptative, varianzaB_adaptative

    func.histograma(sub_imgB_adaptative, "B Despues de filtro adaptativo")

    # Grafico resultados
    func.graficar(imgB, 255, 0, 'gray', 'Original B')
    func.graficar(sub_imgB, 255, 0, 'gray', 'zona homogenea de B')
    func.graficar(imgB_midPoint, 255, 0, 'gray', 'B con filtro de punto medio')
    func.graficar(imgB_adaptative, 255, 0, 'gray', 'B con filtro adaptativo')


def ej6_C():
    # Analisis de C
    imgC = cv.imread("../img/FAMILIA_c.jpg", cv.IMREAD_GRAYSCALE)
    sub_imgC = sub_image("../img/FAMILIA_c.jpg")

    # Analizo el histograma de cada sub imagen
    func.histograma(sub_imgC, "Histograma C")

    # Calculo los parametros estadisticos para cada sub imagen
    mediaC = np.mean(sub_imgC)
    varianzaC = np.var(sub_imgC)
    print mediaC, varianzaC

    # La imagen C tiene ruido sal y pimienta, y ruido gausiano
    sub_imgC_Mediana5x5 = cv.medianBlur(sub_imgC, 5)
    imgCMediana5x5 = cv.medianBlur(imgC, 5)

    mediaC_Mediana5x5 = np.mean(sub_imgC_Mediana5x5)
    varianzaC_Mediana5x5 = np.var(sub_imgC_Mediana5x5)
    print 'mediana 5x5'
    print mediaC_Mediana5x5, varianzaC_Mediana5x5

    func.histograma(sub_imgC_Mediana5x5, "C Despues de mediana 5x5")

    # Mediana 9x9
    sub_imgC_Mediana9x9 = cv.medianBlur(sub_imgC, 9)
    imgCMediana9x9 = cv.medianBlur(imgC, 9)

    mediaC_Mediana9x9 = np.mean(sub_imgC_Mediana9x9)
    varianzaC_Mediana9x9 = np.var(sub_imgC_Mediana9x9)
    print 'mediana 9x9'
    print mediaC_Mediana9x9, varianzaC_Mediana9x9

    func.histograma(sub_imgC_Mediana9x9, "C Despues de mediana 9x9")

    # Grafico resultados
    func.graficar(imgC, 255, 0, 'gray', 'Original C')
    func.graficar(sub_imgC, 255, 0, 'gray', 'Zona homogenea de C')
    func.graficar(imgCMediana5x5, 255, 0, 'gray', 'C despues de mediana 5x5')
    func.graficar(imgCMediana9x9, 255, 0, 'gray', 'C despues de mediana 9x9')

if __name__ == '__main__':
    ej6_C()
    plt.show()

