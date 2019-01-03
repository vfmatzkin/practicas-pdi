# -*- coding: utf-8 -*-

from matplotlib import pyplot as plt
from funciones import *
import itertools

# 1.1 Implemente una LUT del mapeo entre la entrada y la salida.
def ej1_1():
    img = cv.imread("../img/huang1.jpg", cv.IMREAD_GRAYSCALE)

    lookup = mapeo_ad(.9, 150)
    img2 = cv.LUT(img, lookup)

    graficar(img2, 255, 0, 'gray', 'Imagen con LUT')


# 1.2 Pruebe la rutina con diferentes juegos de coeficientes a y c, sobre diversas
# imágenes, y muestre en una misma ventana la imagen original, el mapeo
# aplicado y la imagen obtenida
# 1.4 Genere diversas LUT con estiramientos y compresiones lineales por tramos
# de la entrada, y pruebe los resultados sobre diversas imágenes.
# Franco: Hago a la vez el ejercicio 1.2 y 1.4, ya que hago un mapeo por tramos
def ej1_2():
    img = cv.imread("../img/huang1.jpg", cv.IMREAD_GRAYSCALE)

    lookup = mapeo_tramos()
    img3 = cv.LUT(img, lookup)

    x = range(256)

    f, (ax1, ax2, ax3) = plt.subplots(1, 3)
    ax1.axis("off")
    ax1.imshow(img, cmap='gray')
    ax2.plot(x, lookup)
    # Separate plots
    f.subplots_adjust(wspace=.3)
    ax3.axis("off")
    ax3.imshow(img3, cmap='gray')
    plt.show()


# 1.3 Implemente el negativo de la imagen de entrada.
def ej1_3():
    img = cv.imread("../img/huang1.jpg", cv.IMREAD_GRAYSCALE)

    lookup = mapeo_negativo()
    img2 = cv.LUT(img, lookup)

    graficar(img2, 255, 0, 'gray', 'Negativo de la imagen de entrada')


# 2.1 Implemente la transformación logarı́tmica s = log(1 + r) y la transformación
# de potencia s = r^γ (c=1).
# 2.2 Realice el procesado sobre la imagen ’rmn.jpg’, utilizando los dos procesos
# por separado.
def ej2_1():
    img = cv.imread("../img/rmn.jpg", cv.IMREAD_GRAYSCALE)

    # Mapeo de potencia
    lookup = mapeo_potencia(2,255.0/pow(255,2))
    print lookup
    img2 = cv.LUT(img, lookup)

    graficar(img2, 255, 0, 'gray', 'Imagen con mapeo de potencia')

    plt.figure()
    plt.plot(xrange(256),lookup)
    plt.title("Mapeo de potencia")

    # Mapeo logaritmico
    lookup = mapeo_logaritmico(80)
    print lookup
    img2 = cv.LUT(img, lookup)

    graficar(img2, 255, 0, 'gray', 'Imagen con mapeo logaritmico')


    plt.figure()
    plt.plot(xrange(256),lookup)
    plt.title("Mapeo logaritmico")
    plt.show()


# 3.1a Suma. Normalice el resultado por el número de imágenes.
def ej3_1a():
    img1 = cv.imread("../img/rostro0.png", cv.IMREAD_GRAYSCALE)
    img2 = cv.imread("../img/rostro2.png", cv.IMREAD_GRAYSCALE)

    alpha = 0.7

    img3 = interpolar(img1, img2, alpha)
    # img3 = operaciones_aritmeticas(img2, img1, "SUMA")

    graficar(img3, 255, 0, 'gray', 'Suma')


# 3.1b Resta: Se prueban las dos alternativas que ofrece la consigna, pero parece que cv.subtract
# da un resultado visualmente mejor, por lo que se incluye (junto a otras operaciones)
def ej3_1b():
    img1 = cv.imread("../img/rostro0.png", cv.IMREAD_GRAYSCALE)
    img2 = cv.imread("../img/rostro2.png", cv.IMREAD_GRAYSCALE)

    # img3 = diferencia(img2, img1)
    img3 = operaciones_aritmeticas(img2, img1, "RESTA")

    graficar(img3, 255, 0, 'gray', 'Resta')


# 3.1c Multiplicación: En esta operación la segunda imagen deberá ser una
# máscara binaria, muy utilizada para la extracción de la región de interés
# (ROI) de una imagen.
def ej3_1c():
    img1 = cv.imread("../img/rostro0.png", cv.IMREAD_GRAYSCALE)
    mask = np.zeros(img1.shape)

    # Defino el rectangulo de la mascara
    mask[60:135, 150:250] = 1

    img2 = img1*mask

    graficar(img2, 255, 0, 'gray', 'Multiplicacion')


# 3.1d División. Se implementa como la multiplicación de una imagen por la
# recı́proca de la otra.
def ej3_1d():
    img1 = cv.imread("../img/rostro0.png", cv.IMREAD_GRAYSCALE)
    img2 = cv.imread("../img/rostro2.png", cv.IMREAD_GRAYSCALE)

    img3 = division(img1, img2)

    graficar(img3, 255, 0, 'gray', 'Division')


# 3.2 A partir de una imagen limpia (sin retocar), genere una versión contaminada
# con ruido de distribución normal, de valor medio 0 y varianza 0.05. Reduzca
# el contenido de ruido de la imagen mediante el promediado de no menos de
# 50 imágenes ruidosas
def ej3_2():
    img1 = cv.imread("../img/rostro0.png", cv.IMREAD_GRAYSCALE)
    [altura, ancho] = img1.shape

    mu = 0
    sigma = 0.05
    cant_img_ruidosas = 50
    suma = np.zeros([altura, ancho])

    # Arreglo de imagenes ruidosas
    img_rs = np.zeros([altura, ancho, cant_img_ruidosas])

    for i in xrange(cant_img_ruidosas):
        img_rs[:, :, i] = generar_ruido(img1, mu, sigma)

    for i in xrange(cant_img_ruidosas):
        suma = suma + img_rs[:, :, i]

    print suma.max()
    suma /= suma.max()
    suma *= 255

    graficar(suma, 255, 0, 'gray', 'Imagen reconstruida (suma de imagenes ruidosas)')


# Ejercicio 4: Rodajas del plano de bits
# 1. Implemente una función que reciba como parámetro una imagen de resolución
# máxima de 8 bits y grafique en una ventana la imagen original y los planos
# correspondientes a los bits 0 a 7. Construya una imagen sólo con la informa-
# ción del plano del bit más significativo, luego construya otra imagen con la
# información de los dos planos más significativos, y ası́ sucesivamente. Calcule
# el error cuadrático medio (ECM) entre cada una de éstas y la original.
def ej4():
    img1 = cv.imread("../img/cameraman.tif", cv.IMREAD_GRAYSCALE)

    # Los planos 0-7 vendrán en el arreglo imgs de la forma imgs[:,:,i], donde i es el plano (0 -> LSB).
    imgs = rodajas(img1)

    # Se pide mostrar todos los planos, sólo muestro el 5
    for i in xrange(0,8):
        plt.figure()
        plt.axis("off")
        plt.imshow(imgs[:, :, i], vmax=1, vmin=0, cmap='gray')
        plt.title("Plano "+str(i))

    # Sumo los planos, los muestro y calculo el error cuadrático medio (si RMS da cero es porque sumo todas las imgs).
    img2 = imgs[:,:,0]+imgs[:,:,1]+imgs[:,:,2]+imgs[:,:,3]+imgs[:,:,4]+imgs[:,:,5]+imgs[:,:,6]+imgs[:,:,7]
    graficar(img2, 255, 0, 'gray', 'Imagen reconstruida')

    # Calculo el error cuadratico medio (RMS) como la raiz del producto de 1/cant_elementos y la diferencia entre las
    # imagenes elevada al cuadrado
    err_cuadm = rootmeansquare(img1,img2)
    print "Error cuadratico medio: ", err_cuadm


# 5: Utilizando las técnicas aprendidas, descubra que objetos no están perceptibles
# en la imagen earth.bmp y realce la imagen de forma que los objetos se vuelvan
# visibles con buen contraste sin realizar modificaciones sustanciales en el resto
# de la imagen.
def ej5_1():
    img1 = cv.imread("../img/earth.bmp", cv.IMREAD_GRAYSCALE)
    original = img1.copy()

    histograma(img1[0:280,435:800], "Histograma de la zona oculta")

    lup = mapeo_tramos()
    img1[0:280,435:800] = cv.LUT(img1[0:280,435:800],lup)
    img1 = cv.LUT(img1,lup)

    graficar(img1, 255, 0, 'gray', 'Imagen con LUT por tramos')
    graficar(original, 255, 0, 'gray', 'Imagen de entrada')


# 6. Al final del proceso de manufactura de placas madres, de marca ASUS modelo
# A7V600, se obtienen dos clases de producto final: A7V600-x y A7V600-SE.
# Implemente un algoritmo, que a partir de una imagen, determine que tipo de
# placa es. Haga uso de las técnicas de realce apendidas y utilice las imágenes
# a7v600-x.gif y a7v600-SE.gif. Adapte el método de forma que contemple el
# reconocimiento de imágenes que han sido afectadas por un ruido aleatorio
# impulsivo (a7v600-x(RImpulsivo).gif y a7v600-SE(RImpulsivo).gif ).
def getMotherboardModel(img): #Funcion auxiliar para el ejecicio 5.2
    mask = np.zeros(img.shape)
    mask[96:145, 200:245] = 1
    masked = mask*img

    # Si vemos las medias, podemos notar que en esa region de interés, a7v600-SE y a7v600-SE(RImpulsivo) poseen una
    # media que varía alrededor de 2.25, mientras que a7v600-X y a7v600-X(RImpulsivo) poseen una media que varía entre
    # 0.7 y 0.9, por lo que se establece 1.1 como punto de corte.
    if masked.mean() > 1.1:
        return "a7v600-SE"
    else:
        return "a7v600-X"


def ej5_2():
    # Con OpenCV no puedo cargar gifs, por lo tanto uso matplotlib.pyplot.imread para leer las imagenes.
    # Hay que tener instalado el paquete "Pillow" para que puede leerlos (no hace falta importarlo)
    img1 = plt.imread("../img/a7v600-SE.gif",cv.IMREAD_GRAYSCALE)
    img1_r = plt.imread("../img/a7v600-SE(RImpulsivo).gif", cv.IMREAD_GRAYSCALE)
    img2 = plt.imread("../img/a7v600-X.gif", cv.IMREAD_GRAYSCALE)
    img2_r = plt.imread("../img/a7v600-X(RImpulsivo).gif", cv.IMREAD_GRAYSCALE)

    # No es necesario hacer un procesamiento previo
    # ret, img3 = cv.threshold(img1, 120, 255, cv.THRESH_BINARY)
    print "Estimación de los modelos de las placas: \n"
    print "Archivo: a7v600-SE.gif. Modelo estimado: ", getMotherboardModel(img1)
    print "Archivo: a7v600-SE(RImpulsivo).gif. Modelo estimado: ", getMotherboardModel(img1_r)
    print "Archivo: a7v600-X.gif. Modelo estimado: ", getMotherboardModel(img2)
    print "Archivo: a7v600-X(RImpulsivo).gif. Modelo estimado: ", getMotherboardModel(img2_r)


# 5.3 En una fábrica de medicamentos se desea implementar un sistema para la
# inspección visual automática de blisters en la lı́nea de empaquetado. La ad-
# quisición de la imagen se realiza en escala de grises mediante una cámara
# CCD fija y bajo condiciones controladas de iluminación, escala y enfoque. El
# objetivo consiste en determinar en cada instante si el blister que está siendo
# analizado se encuentra incompleto, en cuyo caso la región correspondiente a
# la pı́ldora faltante presenta una intensidad similar al fondo. Escriba una fun-
# ción que reciba como parámetro la imagen del blister a analizar y devuelva
# un mensaje indicando si el mismo contiene o no la totalidad de las pı́ldo-
# ras. En caso de estar incompleto, indique la posición (x,y) de las pı́ldoras
# faltantes. Verifique el funcionamiento con las imágenes blister completo.jpg y
# blister incompleto.jpg.
def ej5_3():
    imgC = cv.imread("../img/blister_completo.jpg", cv.IMREAD_GRAYSCALE)
    imgI = cv.imread("../img/blister_incompleto.jpg", cv.IMREAD_GRAYSCALE)
    graficar(imgC, 255, 0, 'gray', 'Blister completo')
    graficar(imgI, 255, 0, 'gray', 'Blister incompleto')

    #Resize
    imgI = cv.resize(imgI, (imgC.shape[1], imgC.shape[0]))

    #Umbralizo las imágenes
    ret, imgC = cv.threshold(imgC, 100, 255, cv.THRESH_BINARY)
    ret, imgI = cv.threshold(imgI, 100, 255, cv.THRESH_BINARY)


    #Aplico XOR entre las imagenes
    res = cv.bitwise_xor(imgC, imgI)

    #Suavizo para despues sacar los bordes que no se desean
    # Aplico un filtro de promediado de 10 x 10
    resSuav = cv.filter2D(res, -1, np.ones((9,9), np.float32) / 81)

    #Umbralizo nuevamente
    ret, resUmb = cv.threshold(resSuav, 100, 255, cv.THRESH_BINARY)

    #Guardo en un vector la suma de cada fila y en otro la suma de cada columna
    sumFilas = np.zeros(resUmb.shape[1])
    sumColumnas = np.zeros(resUmb.shape[0])

    sumaC = np.sum(resUmb, axis=0)
    sumaF = np.sum(resUmb, axis=1)

    # contamos donde la intensidad es cero en una linea horizontal
    posCerosC = [i for i, x in enumerate(sumaC) if x == 0]
    posCerosF = [i for i, x in enumerate(sumaF) if x == 0]

    # calculamos los centros entre dos espacios negros
    centroC = []
    for i, val in enumerate(posCerosC):
        if i == 0:
            continue
        else:
            if posCerosC[i] != (posCerosC[i-1] + 1):
                centroC.append((posCerosC[i] + posCerosC[i-1])/2)

    centroF = []
    for i, val in enumerate(posCerosF):
        if i == 0:
            continue
        else:
            if posCerosF[i] != (posCerosF[i-1] + 1):
                centroF.append((posCerosF[i] + posCerosF[i-1])/2)

    # Ahora se calculan todas las permutaciones posibles entre centroides
    permut = list(itertools.product(centroF, centroC))

    for punto in permut:
        if np.mean(resUmb[(punto[0]-5):(punto[0]+5), (punto[1]-5):(punto[1]+5)]) > 0:
            print "Pastilla faltante en: x=",punto[1],", y=",punto[0]

    graficar(imgC, 255, 0, 'gray', 'Blister completo')
    graficar(imgI, 255, 0, 'gray', 'Blister incompleto')
    graficar(res, 255, 0, 'gray', 'Xor')
    graficar(resSuav, 255, 0, 'gray', 'Suavizado')
    # resUmb[centroF[0],centroC[0]] = 0  # Ver punto negro sobre la pastilla faltante (todo borrar)
    graficar(resUmb, 255, 0, 'gray', 'Suavizado binarizado')


# 5.4: Implemente una función que permita “esconder” una imagen binaria en una
# imagen de grises sin que ésto sea percibido a simple vista. Luego, implemente
# una función que permita extraer la imagen binaria. Analice su desempeño.
def ej5_4():
    img1 = cv.imread("../img/cameraman.tif", cv.IMREAD_GRAYSCALE)
    # Los planos 0-7 vendrán en el arreglo imgs de la forma imgs[:,:,i], donde i es el plano (0 -> LSB).
    imgs = rodajas(img1)

    # Notar que esta imagen debe ser mas chica que img1
    imagen_escondida = cv.imread("../img/hidden.png", cv.IMREAD_GRAYSCALE)
    [alto, ancho] = imagen_escondida.shape

    # Pego el mensaje en el plano 0
    imgs[0:alto, 0:ancho, 0] = imagen_escondida/255

    # Muestro el plano modificado
    graficar(imgs[:, :, 0], 1, 0, 'gray', "plano modificado")

    # Rearmo la imagen con el primer plano modificado y la muestro (no deberia notarse el mensaje)
    img2 = imgs[:,:,0]+imgs[:,:,1]+imgs[:,:,2]+imgs[:,:,3]+imgs[:,:,4]+imgs[:,:,5]+imgs[:,:,6]+imgs[:,:,7]
    img2 = img2.astype("uint8")
    graficar(img2, 255, 0, 'gray', "imagen reconstruida con mensaje oculto")

    # Decodificar la imagen con el mensaje
    imgs2 = rodajas(img2)

    # Muestro el plano modificado
    graficar(imgs2[:,:,0], 1, 0, 'gray', "plano 0 de la imagen reconstruida")


if __name__ == '__main__':
    ej1_1()
    plt.show()