# -*- coding: utf-8 -*-

import numpy as np
import math
import cv2 as cv
from matplotlib import pyplot as plt


def mapeo_negativo():
    lookup = np.zeros(256)

    for i in xrange(256):
        lookup[i] = 255 - i

    return lookup


def mapeo_ad(a, d):
    lookup = np.zeros(256)

    for i in xrange(256):
        lookup[i] = i * a + d
        lookup[i] = max(lookup[i], 0)
        lookup[i] = min(lookup[i], 255)
    return lookup


def mapeo_tramos():
    lookup = np.zeros(256)

    for i in range(256):
        if i>0 and i<255:
            lookup[i] = i * 10
        else:
            lookup[i] = i
        lookup[i] = max(lookup[i], 0)
        lookup[i] = min(lookup[i], 255)
    return lookup


def mapeo_potencia(exp,c):
    # Mapeo de potencia: eleva a la exp y multiplica por c
    lookup = np.zeros(256)

    for i in xrange(256):
        lookup[i] = c * pow(i, exp)
        lookup[i] = max(lookup[i], 0)
        lookup[i] = min(lookup[i], 255)
    return lookup


def filterHomomorfico(rows, cols, corte, gL, gH, order):
    """Filtro de magnitud homomorfico"""
    magnitud = np.zeros((rows, cols))
    corte *= rows
    for k in range(rows):
        for l in range(cols):
            d2 = dist([k,l], [rows//2, cols//2])
            magnitud[k,l] = (gH - gL)*(1 - np.exp(-order*(d2/(corte**2)))) + gL
    return np.fft.ifftshift(magnitud)


def dist(a, b):
    """Distancia Euclidea"""
    return np.linalg.norm(np.array(a) - np.array(b))


def mapeo_logaritmico(c):
    lookup = np.zeros(256)

    for i in xrange(256):
        lookup[i] = c * math.log(1 + i)
    lookup = 255 * lookup / lookup.max()
    return lookup


def mapeo_exp(c = 2):
    # Mapeo exponencial: hace e elevado al elemento de la LUT
    lookup = np.zeros(256)

    for i in xrange(256):
        lookup[i] = c * math.exp(i)
    lookup = 255 * lookup / lookup.max()
    return lookup


def interpolar(img1, img2, alpha):
    img3 = img1 * alpha + img2 * (1-alpha)
    return img3


# Se prueban las dos alternativas que ofrece la consigna, pero parece que cv.subtract
# da un resultado visualmente mejor, por lo que se incluye (junto a otras operaciones)
def diferencia(img1, img2):
    img3 = img1 - img2

    # img3 -= img3.min()
    # img3 *= 255/img3.max()

    img3 += 255
    img3 /= 2
    return img3


def operaciones_aritmeticas(img1, img2, tipo):
    TYPES = {
        "SUMA": cv.add(img1, img2),
        "RESTA": cv.subtract(img1, img2),
        "DIVISION": cv.divide(img1, img2),
        "MULTIPLICACION": cv.multiply(img1, img2),
    }

    return TYPES[tipo]


def division(img1, img2):
    img2 = cv.LUT(img2, mapeo_negativo())
    img3 = img1*img2
    img3 /= img3.max()
    img3 *= 255
    return img3


def generar_ruido(img, mu, sigma):
    # img: imagen a la cual se le agrega ruido
    # mu: media
    # sigma: desviacion estandar
    [alto, ancho] = img.shape
    img_re = cv.normalize(img, 0, 1, norm_type=cv.NORM_MINMAX, dtype=cv.CV_64F)
    ruido = np.random.normal(mu, sigma, [alto, ancho]).astype('f')
    img_r = img_re + ruido
    img_r = cv.normalize(img_r, 0, 1, norm_type=cv.NORM_MINMAX, dtype=cv.CV_64F)
    return img_r


def rodajas(img):
    [altura, ancho] = img.shape

    auximg = img*0
    imgs = np.zeros([altura, ancho, 8])

    for i in xrange(8):
        mask = 2**i
        cv.bitwise_and(img, mask, auximg)
        imgs[:, :, i] = np.copy(auximg).astype("uint8")

    return imgs


# Error cuadratico medio entre dos imagenes
def rootmeansquare(img1,img2):
    elems = img1.shape[0] + img1.shape[1]
    RMS = math.sqrt(1 / float(elems) * sum(sum(pow((img2 - img1), 2))))
    return RMS


# A partir de una linea horizontal (altura), calcula en donde se ubican las líneas verticales que separan blanco de negro.
# Ejemplo: Si la imagen es un blister de medicamentos en donde las pastillas son blancas y el fondo negro, al pasarle una
# altura = 55, se fijará en esa línea horizontal, en qué posiciones de X están las pastillas (blanco)
def obtenerPosDesdeAltura(img,altura):
    # perfil de intensidad para esa altura
    perfilh = img[altura, :]
    ceros = []
    for index, valor in enumerate(perfilh):
        if valor == 0:
            ceros.append(index)
    ceros.append(img.shape[1])
    centros = []

    # calculamos los centros entre dos espacios negros
    for i, val in enumerate(ceros):
        if i == 0:
            continue
        else:
            if ceros[i] != (ceros[i - 1] + 1):
                centros.append((ceros[i] + ceros[i - 1]) / 2)

    return centros


def graficar(img,maximo,minimo,mapa,titulo=''):
    ventana = plt.figure()
    # ventana.canvas.set_window_title(titulo)
    plt.axis("off")
    plt.imshow(img, vmax=maximo, vmin=minimo, cmap=mapa)
    plt.title(titulo)


def histograma(img, title="Histograma"):
    plt.figure()
    plt.hist(img.flatten(), 255)
    plt.title(title)


# Elimino los valores que son sucesivos ej:
# Si delFL es True elimino primer y ultimo elemento: [1, 2, 3, 10, 12, 13, 14, 15] ->
# Si delFL es False: [1, 2, 3, 10, 12, 13, 14, 15] -> [1, 3, 10, 12, 15]
def eliminar_contiguos(lista, delFL=False):
    lista2 = []
    # hasta el penultimo elemento
    for i, valor in enumerate(lista[:-1]):
        if lista[i] + 1 != lista[i + 1]:
            lista2.append(lista[i])
            lista2.append(lista[i + 1])

    if not delFL:
        lista2.insert(0, lista[0])
        lista2.append(lista[-1])
    return sorted(list(set(lista2)))

def ruidoSalPimienta(img, s_vs_p, cantidad):
    # Parametros de entrada
    # img: imagen
    # s_vs_p: relacion de sal y pimienta (0 a 1)
    # cantidad: cantidad de ruido

    # Funcion para ensuciar una imagen con ruido sal y pimienta
    (alto, ancho) = img.shape
    # generar ruido tipo sal
    n_sal = np.ceil(cantidad * img.size * s_vs_p)
    coords = [np.random.randint(0, i - 1, int(n_sal)) for i in img.shape]
    img[coords] = 255
    # generar ruido tipo pimienta
    n_pim = np.ceil(cantidad * img.size * (1.0 - s_vs_p))
    coords = [np.random.randint(0, i - 1, int(n_pim)) for i in img.shape]
    img[coords] = 0
    return img

def ruidoGaussiano(img, mu, sigma):
    # img: imagen a la cual se le agrega ruido
    # mu: media
    # sigma: desviacion estandar
    [alto, ancho] = img.shape
    img = img.astype(np.float16) / 255
    ruido = np.random.normal(mu, sigma, [alto, ancho]).astype('f')
    img_r = img + ruido
    img_r = 255 * img_r / img_r.max()
    return img_r

def ruidoRayleigh(img, a):
    (alto, ancho) = img.shape
    img = img.astype(np.float16) / 255
    ruido = np.random.rayleigh(a, [alto, ancho]).astype('f')
    img_r = img + ruido
    img_r = 255 * img_r / img_r.max()
    return img_r

def ruidoUniforme(img, a, b):
    (alto, ancho) = img.shape
    img = img.astype(np.float16) / 255
    ruido = np.random.uniform(a, b, [alto, ancho]).astype('f')
    img_r = img + ruido
    img_r = 255 * img_r / img_r.max()
    return img_r

def ruidoExponencial(img, a):
    (alto, ancho) = img.shape
    img = img.astype(np.float16) / 255
    ruido = np.random.exponential(a, [alto, ancho]).astype('f')
    img_r = img + ruido
    img_r = 255 * img_r / img_r.max()
    return img_r

def ruidoGamma(img, a, b):
    (alto, ancho) = img.shape
    img = img.astype(np.float16) / 255
    ruido = np.random.gamma(a, b, [alto, ancho]).astype('f')
    img_r = img + ruido
    img_r = 255 * img_r / img_r.max()
    return img_r

def filtroMediaGeometrica(img, m, n):
    (s, t) = img.shape
    for i in range(0, s-m+1):
        for j in range(0, t-n+1):
            acum = 1
            for k in range(i, i+m):
                for o in range(j, j+n):
                    acum = acum * img[k, o]
            img[i,j] = float(pow(acum, 1.0/(m*n)))
    return img

def filtroMediaContraarmonica(img, Q, s, t):
    # Si Q vale 0 da la media aritmética (Q > 0 elimina pimienta)
    # Si Q vale -1 va la media armónica (elimina sal)
    (m, n) = img.shape
    for i in range(0, m-s+1):
        for j in range(0, n-t+1):
            cont1 = 0
            cont2 = 0
            for k in range(i, i+s):
                for o in range(j, j+t):
                    cont1 = cont1 + np.power(img[k, o], Q+1)
                    cont2 = cont2 + np.power(img[k, o], Q)
            img[i, j] = cont1 / cont2
    return img

def ECM(img1, img2):
    # Error cuadrático medio (Root mean square)
    img = img1 - img2
    return math.sqrt(sum(n * n for n in img.flatten()) / len(img.flatten()))

def capturar_punto(event, x, y, flags, param):
    global mouseX,mouseY, imgn
    if event == cv.EVENT_LBUTTONDOWN:
            print "posicion elegida: (",x,",",y,"), presione 'a' o 'c' para confirmar."
            mouseX,mouseY = x,y

def elegir_punto(image):
    global imgn
    imgn = image
    cv.namedWindow("image")
    cv.setMouseCallback("image", capturar_punto)

    # keep looping until the 'q' key is pressed
    while True:
        # display the image and wait for a keypress
        cv.imshow("image", imgn)
        key = cv.waitKey(20) & 0xFF

        if key == 27:
            break
        elif key == ord('a'):
            return [mouseX, mouseY]
        elif key == ord('c'):
            return [mouseX, mouseY]

def filter_midPoint3(source):
    # Filtro de punto medio de 3 x 3
    # Se le pasa la 1 canal de la imagen
    # Devuelve la imagen filtrada

    final = source.copy().astype(np.uint16)
    rows = source.shape[0]
    cols = source.shape[1]
    members = [source[0, 0]] * 9
    for y in range(1, rows - 1):
        for x in range(1, cols - 1):
            members[0] = source[y - 1, x - 1]
            members[1] = source[y, x - 1]
            members[2] = source[y + 1, x - 1]
            members[3] = source[y - 1, x]
            members[4] = source[y, x]
            members[5] = source[y + 1, x]
            members[6] = source[y - 1, x + 1]
            members[7] = source[y, x + 1]
            members[8] = source[y + 1, x + 1]

            a = max(members)
            b = min(members)
            c = (a.astype(int) + b.astype(int)) / 2
            final[y, x] = c

    final.astype(np.uint8)
    return final

def filter_adaptative(source, varRuido):
    # Filtro adaptativo de 3 x 3
    # Se le pasa la 1 canal de la imagen
    # Devuelve la imagen filtrada

    final = source.copy().astype(np.uint8)
    rows = source.shape[0]
    cols = source.shape[1]
    members = [source[0, 0]] * 9
    for y in range(1, rows - 1):
        for x in range(1, cols - 1):
            members[0] = source[y - 1, x - 1]
            members[1] = source[y, x - 1]
            members[2] = source[y + 1, x - 1]
            members[3] = source[y - 1, x]
            members[4] = source[y, x]
            members[5] = source[y + 1, x]
            members[6] = source[y - 1, x + 1]
            members[7] = source[y, x + 1]
            members[8] = source[y + 1, x + 1]

            media = np.mean(members)
            varLocal = np.std(members)

            if int(varLocal) is not 0:
                aux = (source[y, x] - (varRuido/varLocal) * (source[y, x]-media))
                aux = min(aux, 255)
                aux = max(aux, 0)
                final[y, x] = int(aux)

    final.astype(np.uint8)
    return final

def bordes_roberts(img, umbral1, umbral2):
    roberts_cross_v = np.array([[0, 0, 0],
                                [0, 1, 0],
                                [0, 0, -1]])

    roberts_cross_h = np.array([[0, 0, 0],
                                [0, 0, 1],
                                [0, -1, 0]])

    horizontal = cv.filter2D(img, -1, roberts_cross_h).astype(np.uint16)
    vertical = cv.filter2D(img, -1, roberts_cross_v).astype(np.uint16)

    output_image = np.sqrt(np.square(horizontal) + np.square(vertical)).astype(np.uint8)

    ret, output_image = cv.threshold(output_image, umbral1, umbral2, cv.THRESH_BINARY)
    return output_image


def bordes_prewitt(img, umbral1, umbral2):
    kernelx = np.array([[1, 1, 1],
                        [0, 0, 0],
                        [-1, -1, -1]])

    kernely = np.array([[-1, 0, 1],
                        [-1, 0, 1],
                        [-1, 0, 1]])
    img_prewittx = cv.filter2D(img, -1, kernelx).astype(np.uint16)
    img_prewitty = cv.filter2D(img, -1, kernely).astype(np.uint16)

    output_image = np.sqrt(np.square(img_prewittx) + np.square(img_prewitty)).astype(np.uint8)
    ret, output_image = cv.threshold(output_image, umbral1, umbral2, cv.THRESH_BINARY)
    return output_image


def bordes_sobel(img, umbral1, umbral2):
    img_sobelx = cv.Sobel(img, cv.CV_16U, 1, 0, ksize=3)
    img_sobely = cv.Sobel(img, cv.CV_16U, 0, 1, ksize=3)
    # output_image = img_sobelx + img_sobely

    output_image = np.sqrt(np.square(img_sobelx) + np.square(img_sobely)).astype(np.uint8)
    ret, output_image = cv.threshold(output_image, umbral1, umbral2, cv.THRESH_BINARY)

    return output_image


def bordes_laplaciano(img, umbral1, umbral2):
    output_image = cv.Laplacian(img, cv.CV_8U)
    ret, output_image = cv.threshold(output_image, umbral1, umbral2, cv.THRESH_BINARY)

    return output_image


def bordes_LoG(img, umbral1, umbral2):
    kernel = np.array([[0, 0, -1, 0, 0],
              [0, -1, -2, -1, 0],
              [-1, -2, 16, -2, -1],
              [0, -1, -2, -1, 0],
              [0, 0, -1, 0, 0]])
    output_image = cv.filter2D(img, -1, kernel)
    ret, output_image = cv.threshold(output_image, umbral1, umbral2, cv.THRESH_BINARY)

    return output_image


def filter_adaptative2(source, varRuido, size):
    final = source.copy().astype(np.uint8)
    cols, rows = source.shape
    for y in range(size/2, rows - size/2):
        for x in range(size/2, cols - size/2):
            x1 = max(0, x-size/2)
            x2 = min(cols+1, x+size/2)
            y1 = max(0, y-size/2)
            y2 = min(rows+1, y+size/2)

            media = np.mean(source[x1:x2,y1:y2])
            varLocal = np.std(source[x1:x2,y1:y2])

            if np.isnan(varLocal):
                print "hola"

            if int(varLocal) is not 0:
                aux = (source[x,y] - (varRuido/varLocal) * (source[x,y]-media))
                aux = min(aux, 255)
                aux = max(aux, 0)
                final[x, y] = int(aux)

    final.astype(np.uint8)
    return final

#todo Verificar en el libro si están bien o al revés
def apertura(A, B, it = 1):
    C = cv.dilate(A,B, iterations = it)
    # func.graficar(C, 255, 0, 'gray', 'Dilatacion')
    D = cv.erode(C,B, iterations = it)

    return D

def cierre(A, B, it = 1):
    C = cv.erode(A,B, iterations = it)
    # func.graficar(C, 255, 0, 'gray', 'Erosion')
    D = cv.dilate(C,B, iterations = it)

    return D

def segmentarColorHS(img, subimg, varF=30):
    rows,cols = img.shape[0:2]
    # Centroides de H y S
    medH = np.mean(subimg[:,:,0])
    medS = np.mean(subimg[:,:,1])

    # Calculo los radios de  H y S como la varianza de cada componente
    rH = np.std(subimg[:,:,0]) * varF
    rS = np.std(subimg[:,:,1]) * varF

    H = img[:, :, 0]
    S = img[:, :, 1]

    # Recorro imgCompleta y veo si esta dentro de los circulos en cada canal
    umbColor = np.zeros((rows, cols), dtype=np.uint8)
    for i in range(rows):
        for j in range(cols):
            if H[i, j] <= medH + rH and H[i, j] >= medH - rH:
                if S[i, j] <= medS + rS and S[i, j] >= medS - rS:
                    umbColor[i, j] = 255
    umbColor = 255 - umbColor
    return umbColor

def segmentarColorSV(img, subimg, var1=15, var2=10):
    rows,cols = img.shape[0:2]
    # Centroides de H y S
    medS = np.mean(subimg[:,:,1])
    medV = np.mean(subimg[:,:,2])

    # Calculo los radios de  H y S como la varianza de cada componente
    rS = np.std(subimg[:,:,1]) * var1
    rV = np.std(subimg[:,:,2]) * var2

    S = img[:, :, 1]
    V = img[:, :, 2]

    # Recorro imgCompleta y veo si esta dentro de los circulos en cada canal
    umbColor = np.zeros((rows, cols), dtype=np.uint8)
    for i in range(rows):
        for j in range(cols):
            if S[i, j] <= medS + rS and S[i, j] >= medS - rS:
                if V[i, j] <= medV + rV and V[i, j] >= medV - rV:
                    umbColor[i, j] = 255
    umbColor = 255 - umbColor
    return umbColor


# Me canse de copiar y pegar siempre lo mismo, asi que va un intento de generalizar:
#   -source: imagen de bordes a la cual se aplica la transformada de Hough
#   -n: Cantidad de lineas a dibujar. Por defecto es 1. Para dibujar todas las lineas pasar -1
#   -rhosensv: Precision de rho en pixeles
#   -phisensv: Precision de phi en radianes
#   -hardness: Solo las lineas con una cantidad mayor a estas seran devueltas
#   -plotimg: Imagen sobre la cual graficar las lineas
#   -color: Color de las lineas a graficar
def hough(source, n = 1, rhosensv = 1, phisensv = np.pi / 180, hardness = 90, plotimg = None, color = (0, 0, 255)):
    rows, cols = source.shape
    lines = cv.HoughLines(source, rhosensv,phisensv, hardness)
    # Gráfica de la primera línea detectada (la más fuerte)
    if n == -1:
        n = lines.shape[0]
    points = []
    for line in lines[0:n]:
        for rho, theta in line:
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a * rho
            y0 = b * rho
            x1 = max(int(x0 + 1000 * (-b)), 0)
            y1 = max(int(y0 + 1000 * (a)), 0)
            x2 = min(int(x0 - 1000 * (-b)), cols)
            y2 = min(int(y0 - 1000 * (a)), rows)
            points.append([(x1, y1), (x2, y2)])
            if plotimg is not None:
                cv.line(plotimg, (x1, y1), (x2, y2), color, 2)
    return lines, points