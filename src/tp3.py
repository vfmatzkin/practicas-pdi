# -*- coding: utf-8 -*-

from matplotlib import pyplot as plt
import cv2 as cv
import numpy as np
from funciones import *
from scipy.stats.stats import pearsonr


# 1.1 Cargue  una  imagen  y  realice  la  ecualizacion  de  su  histograma.  Muestre  en
# una misma ventana la imagen original, la version ecualizada y sus respectivos
# histogramas y estudie la informacion suministrada por el histograma. Repita
# el analisis para distintas imagenes
def ej1_1():
    img = cv.imread("../img/imagenB.tif", cv.IMREAD_GRAYSCALE)

    # Imagen original
    graficar(img, 255, 0, 'gray', 'Imagen de entrada')

    # Calcular histograma
    histograma(img)

    # Ecualizar imagen
    imgeq = cv.equalizeHist(img)

    # Imagen ecualizada
    graficar(imgeq, 255, 0, 'gray', 'Imagen ecualizada')
    histograma(imgeq, 'Histograma de la imagen ecualizada')


# 1.2 Los archivos histo1.tif, histo2.tif, histo3.tif, histo4.tif e histo5.tif
# contienen histogramas de imagenes con diferentes caracterısticas. Se pide:
# Analizando solamente los archivos de histogramas, realice una descripcion
# de la imagen (es clara u oscura?, tiene buen contraste?, etc.).
# Anote la correspondencia histograma-imagen con los archivos imagenA.tif a
# imagenE.tif. Cargue las imagenes originales y muestre los histogramas. Compare con
# sus respuestas del punto anterior.

# histo1.tif: La imagen tiene mucho oscuro pero tambien tiene partes claras y alto contraste
# histo2.tif: La imagen es predominantemente gris oscuro, con un contraste pequeño
# histo3.tif: La imagen es muy oscura, de poco contraste
# histo4.tif: La imagen es muy clara, con poco contraste
# histo5.tif: La imagen tiene pocos colores oscuros, con un contraste mediano.

# imagenA.tif: histo2.tif
# imagenB.tif: histo4.tif
# imagenC.tif: histo1.tif
# imagenD.tif: histo5.tif
# imagenE.tif: histo3.tif

# 2.1 Genere diferentes mascaras de promediado. Aplique los filtros sobre una ima-
# gen y verifique los efectos del aumento del tamano de la mascara en la imagen
# resultante.
def ej2_1():
    img = cv.imread("../img/cameraman.tif", cv.IMREAD_GRAYSCALE)
    kernel1 = np.ones((5,5), np.float32)/25
    kernel2 = np.ones((10,10), np.float32)/100
    kernel3 = np.array([[0, 0, 0], [-1, 2, -1], [0, 0, 0]])

    dst1 = cv.filter2D(img,-1, kernel1)
    dst2 = cv.filter2D(img,-1, kernel2)
    dst3 = cv.filter2D(img,-1, kernel3)

    graficar(dst1, 255, 0, 'gray', 'Filtro cuadrado 5x5')
    graficar(dst2, 255, 0, 'gray', 'Filtro cuadrado 10x10')
    graficar(dst3, 255, 0, 'gray', 'Filtro horizontal suma cero')


# 2.2 Genere mascaras de filtrado gaussianas con diferente
# σ y diferente tamano.
# Visualice y aplique las mascaras sobre una imagen. Compare los resultados
# con los de un filtro de promediado del mismo tamaño
def ej2_2():
    img = cv.imread("../img/letras1.tif", cv.IMREAD_GRAYSCALE)
    blur = cv.GaussianBlur(img, (25,25), 0)
    prom = cv.filter2D(img,-1,np.ones((25,25),np.float32)/(25*25))

    graficar(img, 255, 0, 'gray', 'Imagen original')
    graficar(blur, 255, 0, 'gray', 'Filtro gaussiano 25x25')
    graficar(prom, 255, 0, 'gray', 'Filtro promediador 25x25')


# 2.3 Los  filtros  pasa-bajos  son  muy utilizados  para localizar objetos  grandes  en
# una escena. Aplique este concepto a la imagen  ’hubble.tif’
# y obtenga una  imagen de grises cuyos objetos correspondan solamente a los de mayor tamano
# de la original.
def ej2_3():
    img = cv.imread("../img/hubble.tif", cv.IMREAD_GRAYSCALE)

    # Aplico un filtro de promediado de 10 x 10
    prom = cv.filter2D(img,-1,np.ones((10,10),np.float32)/100)
    # Umbral en 150
    ret, img3 = cv.threshold(prom, 150, 255, cv.THRESH_BINARY)
    # Divido por 255 para poder multiplicar
    img3 /= 255

    graficar(img, 255, 0, 'gray', 'Imagen original')
    graficar(img3, 1, 0, 'gray', 'Filtro 10x10 + Umbral en 150')
    graficar(img3*img, 255, 0, 'gray', 'Multiplicacion')
    plt.show()


# 3.1 Defina mascaras de filtrado pasa-altos cuyos coeficientes sumen 1 y aplıquelas
# sobre diferentes imagenes. Interprete los resultados.
def ej3_1():
    img = cv.imread("../img/huang2.jpg", cv.IMREAD_GRAYSCALE)
    kernel1 = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    kernel2 = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])

    filtered = cv.filter2D(img,-1,kernel1)
    filtered2 = cv.filter2D(img,-1,kernel2)
    # Se puede ver que las altas frecuencias se realzan sin alterar a las bajas
    graficar(filtered, 255, 0, 'gray', 'Imagen filtrada')
    graficar(filtered2, 255, 0, 'gray', 'Imagen filtrada')


# 3.2 Repita el ejercicio anterior para mascaras cuyos coeficientes sumen 0. Com-
# pare los resultados con los del punto anterior.
def ej3_2():
    img = cv.imread("../img/huang2.jpg", cv.IMREAD_GRAYSCALE)
    kernel1 = np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]])
    kernel2 = np.array([[-5, -1, 5], [-2, 6, -2], [1, -1, -1]])

    filtered = cv.filter2D(img,-1,kernel1)
    filtered2 = cv.filter2D(img,-1,kernel2)
    # Se puede ver que se extraen las altas frecuencias eliminando las bajas (zonas homogeneas).
    graficar(filtered, 255, 0, 'gray', 'Imagen filtrada con [[0, -1, 0], [-1, 4, -1], [0, -1, 0]]')
    graficar(filtered2, 255, 0, 'gray', 'Imagen filtrada con [[-5, -1, 5], [-2, 6, -2], [1, -1, -1]]')


#  4.1 Obtenga versiones mejoradas de diferentes imagenes mediante el filtrado por
# mascara difusa. Implemente el calculo como f(x,y)−PB(f(x,y)).
def ej4_1():
    img = cv.imread("../img/camino.tif", cv.IMREAD_GRAYSCALE)
    kernel1 = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])

    filtered = cv.filter2D(img,-1,kernel1)

    img2 = cv.subtract(img,filtered)

    # Se puede ver que se extraen las altas frecuencias eliminando las bajas (zonas homogeneas).
    graficar(img, 255, 0, 'gray', 'Imagen original')
    graficar(img2, 255, 0, 'gray', 'Imagen realzada')


#  4.2 Una forma de enfatizar las altas frecuencias sin perder los detalles de bajas
# frecuencias  es  el  filtrado  de  alta  potencia.  Implemente  este  procesamiento
# como  la  operacion  aritmetica: f = Af(x,y)−PB(f(x,y)),  con A ≥ 1. Investigue y pruebe metodos
# alternativos de calculo en una pasada.
def ej4_2():
    img = cv.imread("../img/camaleon.tif", cv.IMREAD_GRAYSCALE)
    n = 5
    filtered = cv.filter2D(img,-1,np.ones((n,n),np.float32)/(n*n))

    A = 2

    img2 = cv.subtract(float(A)*img, 1.0*filtered)
    graficar(img, 255, 0, 'gray', 'Imagen original')
    graficar(img2, 255, 0, 'gray', 'Imagen realzada')


# 5.1 En la imagen cuadros.tif se observa un conjunto de cuadros negros sobre un
# fondo casi uniforme. Utilice ecualización local del histograma para revelar
# los detalles ocultos en la imagen y compare los resultados con los obtenidos
# con ecualización global. Ayuda: la clave es el tamaño de ventana para la
# ecualización local. También se debe determinar donde aplicar la ecualización
# local y donde no.
def ej5_1():
    img = cv.imread("../img/cuadros.tif", cv.IMREAD_GRAYSCALE)

    # sumo los valores en cada fila y cada columna
    filas = [sum(fil) for fil in np.array(img)]
    columnas = [sum(col) for col in np.transpose(img)]

    # obtengo el valor maximo, que representa en que posiciones estará el
    # fondo, ya que es más claro que lo que buscamos
    maxval = np.array(filas).max()
    # para buscarlo, se le da un margen de 1000 ya que tiene leves variaciones
    maxval -= 1000
    # busco en que posiciones esta el fondo
    posfondof = [i for i, x in enumerate(filas) if x > maxval]
    posfondoc = [i for i, x in enumerate(columnas) if x > maxval]

    # Elimino los numeros consecutivos: [1, 2, 3, 10, 12, 13, 14, 15] -> [3, 10, 12]
    posfondoc = eliminar_contiguos(posfondoc, True)
    posfondof = eliminar_contiguos(posfondof, True)
    print posfondoc
    original = img.copy()
    img[posfondof[0]:posfondof[1], posfondoc[0]:posfondoc[1]] = cv.equalizeHist(img[posfondof[0]:posfondof[1],posfondoc[0]:posfondoc[1]])
    img[posfondof[0]:posfondof[1], posfondoc[4]:posfondoc[5]] = cv.equalizeHist(img[posfondof[0]:posfondof[1],posfondoc[4]:posfondoc[5]])
    img[posfondof[2]:posfondof[3], posfondoc[2]:posfondoc[3]] = cv.equalizeHist(img[posfondof[2]:posfondof[3],posfondoc[2]:posfondoc[3]])
    img[posfondof[4]:posfondof[5], posfondoc[0]:posfondoc[1]] = cv.equalizeHist(img[posfondof[4]:posfondof[5],posfondoc[0]:posfondoc[1]])
    img[posfondof[4]:posfondof[5], posfondoc[4]:posfondoc[5]] = cv.equalizeHist(img[posfondof[4]:posfondof[5],posfondoc[4]:posfondoc[5]])

    graficar(original, 255, 0, 'gray', 'Imagen original')
    graficar(img, 255, 0, 'gray', 'Imagen ecualizada por partes')


# Proponga una combinación de técnicas para realzar los detalles de la imagen
# esqueleto.tif. Justifique cada una de las elecciones en la elaboración de su
# propuesta.
def ej5_2():
    img = cv.imread("../img/esqueleto.tif", cv.IMREAD_GRAYSCALE)
    graficar(img, 255, 0, 'gray', 'Imagen original')

    median = cv.medianBlur(img, 5)
    graficar(median, 255, 0, 'gray', 'Filtro de mediana 5x5')

    n = 15
    filtered = cv.filter2D(median,-1,np.ones((n,n),np.float32)/(n*n))

    A = 1.5

    img2 = cv.subtract(float(A)*median, 1.0*filtered)

    # # Se puede ver que se extraen las altas frecuencias eliminando las bajas (zonas homogeneas).
    graficar(img2, 255, 0, 'gray', 'Mediana 5x5 -> High boost A=2 5x5')


def ej5_3():
    ruta = "../img/Bandera05.jpg"

    #Abro una imagen de cada tipo y calculo el histograma
    bandera = cv.imread("../img/Bandera01.jpg", cv.IMREAD_GRAYSCALE)
    histBandera = cv.calcHist([bandera],[0],None,[256],[0,256])

    caricatura = cv.imread("../img/Caricaturas01.jpg", cv.IMREAD_GRAYSCALE)
    histCaricatura = cv.calcHist([caricatura],[0],None,[256],[0,256])

    paisaje = cv.imread("../img/Paisaje01.JPG", cv.IMREAD_GRAYSCALE)
    histPaisaje = cv.calcHist([paisaje],[0],None,[256],[0,256])

    personaje = cv.imread("../img/Personaje01.jpg", cv.IMREAD_GRAYSCALE)
    histPersonaje = cv.calcHist([personaje],[0],None,[256],[0,256])

    prueba = cv.imread(ruta, cv.IMREAD_GRAYSCALE)
    histPrueba = cv.calcHist([prueba], [0], None, [256], [0, 256])

    #Calculo la correlacion de los histogramas
    coefB = pearsonr(histBandera, histPrueba)
    coefC = pearsonr(histCaricatura, histPrueba)
    coefPai = pearsonr(histPaisaje, histPrueba)
    coefPer = pearsonr(histPersonaje, histPrueba)

    print "Coeficientes de bandera, caricatura, paisaje, personaje: "
    print coefB[0], coefC[0], coefPai[0], coefPer[0]


if __name__ == '__main__':
    ej5_3()
    plt.show()
