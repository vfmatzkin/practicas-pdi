from __future__ import division
import cv2 as cv
import numpy as np
import math as ma
import matplotlib.pyplot as plt

####################### COMANDOS UTILES #######################
# Redimensionar una imagen
#   img = cv.resize(img,(0, 0),None,.25,.25)

# Pasar de una imagen BGR a RGB, o viceversa:
#   img = img[:,:,::-1]

# Convertir una imagen BGR a HSV, o viceversa:
#   imghsv = cv.cvtColor(img1,cv.COLOR_BGR2HSV)
#   imghsv = cv.cvtColor(img1,cv.COLOR_HSV2BGR)

# Crear un kernel para filtros:
# kernel = np.array(np.mat('-1 -1 -1; -1 9 -1; -1 -1 -1'))

# Aplicacion individual de un kernel a un canal i:
# img[:,:,i] = cv.filter2D(img[:,:,i],-1,kernel)

# Ecualizacion individual de histogramas para un canal i:
# img[:,:,i] = cv.equalizeHist(img[:,:,i])

# Ver el histograma de un canal individual:
# plt.hist(img[:,:,i].ravel(),256,[0,256])

################################################################

def imagenesVentana(nombre,imgs,orientacion):
    """ Muestra varias imagenes en una sola ventana """
    if len(imgs)==8:
        concat1 = np.concatenate(imgs[0:4], axis=1)
        concat2 = np.concatenate(imgs[4:8], axis=1)
        concat = np.concatenate([concat1,concat2], axis=0)
    else:
        concat = np.concatenate(imgs, axis=orientacion)
    cv.imshow(nombre, concat)
    cv.waitKey()
    cv.destroyAllWindows()
    return concat

def plotAsImage(s,name):
    """ Guarda el grafico como una imagen y la muestra en OpenCV """
    r = range(0,256)
    grafico = plt.figure()
    plt.plot(r,s)
    plt.xlim(0,255)
    plt.ylim(0,265)
    grafico.savefig(name)
    img = cv.imread(name,cv.IMREAD_GRAYSCALE)
    return img

def histToImg(img,name):
    grafico = plt.figure()
    plt.hist(img.flatten(),256,[0,256])
    plt.xlim([0,256])
    grafico.savefig(name)
    newimg = cv.imread(name,cv.IMREAD_COLOR)
    return newimg

def transformacionLineal(a,c):
    """ LUT de acuerdo a los valores de a (ganancia) y c (offset):
        Si a=1 => Manejo del offset
        Si c=0 => Amplificacion """
    s = []
    for r in range(0,256):
        t = a*r + c
        s.append(max(min(t,255),0))
    return s

def negativo(rmax):
    """ LUT de negativo de una imagen """
    s = []
    for r in range(0,256):
        t = rmax-r
        s.append(max(min(t,rmax),0))
    return s

def binarizacion(p,smax,b):
    """ LUT para binarizar una imagen, donde p es el umbral:
        Si b=1 => de 0 a p pasa a ser negro
        Si b=0 => de 0 a p pasa a ser blanco """
    # b es una bandera para elegir entre una binarizacion y su inversa
    # b=1 es la binarizacion y b=0 es la inversa
    s = np.zeros(256)
    if b==1:
        for r in range(p+1,256):
            s[r] = smax
    else:
        for r in range(0,p):
            s[r] = smax
    return s

def umbralIntervalo(p1,p2,smax,b):
    """ LUT para binarizar una imagen por intervalos p1 y p2:
        Si b=1 => de 0 a p1 y de p2 a 255 pasa a ser blanco
        Si b=0 => de p1 a p2 pasa a ser blanco """
    s = np.zeros(256)
    if b==1:
        for r in range(0,p1+1):
            s[r] = smax
        for r in range(p2,256):
            s[r] = smax
    else:
        for r in range(p1+1,p2):
            s[r] = smax
    return s

def umbralEscalaGrises(p1,p2,smax,b):
    s = np.zeros(256)
    for r in range(0, p1 + 1):
        s[r] = smax
    for r in range(p2, 256):
        s[r] = smax
    if b==1:
        for r in range(p1 + 1, p2):
            s[r] = r
    else:
        for r in range(p1 + 1, p2):
            s[r] = smax - r
    return s

def transformacionLog():
    s = []
    c = 255.0/ma.log10(1+255)
    for r in range(0,256):
        t = c*ma.log10(1+r)
        s.append(max(min(t,255),0))
    return s

def transformacionPotencia(gamma):
    s = []
    for r in range(0,256):
        t = ma.pow(r/255.0,gamma)*255.0
        s.append(max(min(t,255),0))
    return s

def rangoDinamico(rlist,slist):
    s = np.zeros(256)
    for i in range(0,len(rlist)-1):
        p1 = [rlist[i],slist[i]]
        p2 = [rlist[i+1],slist[i+1]]
        a = float(p2[1]-p1[1])/(p2[0]-p1[0])
        b = p1[1] - a*p1[0]
        for j in range(p1[0],p2[0]+1):
            y = a*j + b
            s[j] = np.clip(y,0,255)
    return s

def aplicarLUT(img,LUT):
    """ Aplica una LUT a una imagen, devolviendo la imagen resultante """
    H, W = img.shape
    newimg = img.copy()
    for i in range(0,H):
        for j in range(0,W):
            newimg[i][j] = LUT[img[i][j]]
    return newimg

def sumaImg(img1,img2):
    H, W = img1.shape
    newimg = img1.copy()
    for i in range(0,H):
        for j in range(0,W):
            newimg[i][j] = (int(img1[i][j]) + int(img2[i][j]))/2
    return newimg

def restaImg(img1,img2):
    H, W = img1.shape
    newimg = img1.copy()
    for i in range(0,H):
        for j in range(0,W):
            newimg[i][j] = int(img1[i][j]) - int(img2[i][j])
            newimg[i][j] += 255
            newimg[i][j] /= 2
    return newimg

def multImg(img1,mask):
    H, W = img1.shape
    newimg = img1.copy()
    for i in range(0,H):
        for j in range(0,W):
            newimg[i][j] = int(img1[i][j]) * int(mask[i][j])
    return newimg

def findBordes(img,line):
    """ Funcion para encontrar bordes en una cierta linea,
    donde la imagen debe estar binarizada"""
    H, W = img.shape
    bordes = []
    for j in range(1,W):
        # Si dos pixeles consecutivos tienen distintos valores, guardo el pixel de interes
        if (img[line][j]==255) != (img[line][j-1]==255):
            if img[line][j] == 0:
                bordes.append(j-1)
            else:
                bordes.append(j)
    return bordes

def ecualizar(histograma):
    cdf = histograma.cumsum()
    cdf_m = np.ma.masked_equal(cdf,0)
    cdf_m = (cdf_m-cdf_m.min())*255/(cdf_m.max()-cdf_m.min())
    cdf = np.ma.filled(cdf_m,0).astype('uint8')
    return cdf

def canalesRGB(img):
    bgr = cv.split(img)
    cv.imshow("Canal R", bgr[2])
    cv.imshow("Canal G", bgr[1])
    cv.imshow("Canal B", bgr[0])
    return

def canalesHSV(img):
    img_hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    hsv = cv.split(img_hsv)
    cv.imshow("Canal H", hsv[0])
    cv.imshow("Canal S", hsv[1])
    cv.imshow("Canal V", hsv[2])
    return

def optimalDFTImg(img):
    """Zero-padding sobre img para alcanzar un tamanio optimo para FFT"""
    h = cv.getOptimalDFTSize(img.shape[0])
    w = cv.getOptimalDFTSize(img.shape[1])
    return cv.copyMakeBorder(img, 0, h - img.shape[0], 0, w - img.shape[1], cv.BORDER_CONSTANT)

def spectrum(img):
    """Calcula y muestra el modulo logartimico de la DFT de img."""
    # img=optimalDFTImg(img)
    imgf = cv.dft(np.float32(img), flags=cv.DFT_COMPLEX_OUTPUT)
    modulo = np.log(cv.magnitude(imgf[:,:,0], imgf[:,:,1]) + 1)
    modulo = np.fft.fftshift(modulo)
    modulo = cv.normalize(modulo, modulo, 0, 1, cv.NORM_MINMAX)
    plt.figure()
    plt.subplot(1,2,1)
    plt.imshow(img, cmap='gray')
    plt.subplot(1,2,2)
    plt.imshow(modulo, cmap='gray')
    plt.show()
    return modulo

def rotate(img, angle):
    """Rotacion de la imagen sobre el centro"""
    r = cv.getRotationMatrix2D((img.shape[0]/2, img.shape[1]/2), angle, 1.0)
    return cv.warpAffine(img, r, img.shape)

def filterImg(img, filtro_magnitud):
    """Filtro para imagenes de un canal"""
    # Como la fase del filtro es 0 la conversion de polar a cartesiano es directa (magnitud->x, fase->y)
    filtro = np.array([filtro_magnitud, np.zeros(filtro_magnitud.shape)]).swapaxes(0, 2).swapaxes(0, 1)
    imgf = cv.dft(np.float32(img), flags=cv.DFT_COMPLEX_OUTPUT)
    imgf = cv.mulSpectrums(imgf, np.float32(filtro), cv.DFT_ROWS)
    return cv.idft(imgf, flags=cv.DFT_REAL_OUTPUT | cv.DFT_SCALE)

def dist(a, b):
    """Distancia Euclidea"""
    return np.linalg.norm(np.array(a) - np.array(b))

def filterGaussian(rows, cols, corte):
    """Filtro de magnitud gausiano"""
    magnitud = np.zeros((rows, cols))
    corte *= rows
    for k in range(rows):
        for l in range(cols):
            magnitud[k,l] = np.exp(-dist([k,l], [rows//2, cols//2])/2/corte)
    return np.fft.ifftshift(magnitud)

def filterGaussianPasaAltos(rows, cols, corte):
    """Filtro de magnitud gausiano"""
    magnitud = np.zeros((rows, cols))
    corte *= rows
    for k in range(rows):
        for l in range(cols):
            magnitud[k,l] = 1 - np.exp(-dist([k,l], [rows//2, cols//2])/2/corte)
    return np.fft.ifftshift(magnitud)

def filterIdeal(rows, cols, corte):
    """filtro de magnitud ideal"""
    magnitud = np.zeros((rows, cols))
    magnitud = cv.circle(magnitud, (cols//2, rows//2), int(rows*corte), 1, -1)
    return np.fft.ifftshift(magnitud)

def filterIdealPasaAltos(rows, cols, corte):
    """filtro de magnitud ideal"""
    magnitud = np.ones((rows, cols))
    magnitud = cv.circle(magnitud, (cols//2, rows//2), int(rows*corte), 0, -1)
    return np.fft.ifftshift(magnitud)

def filterButterworth(rows, cols, corte, order):
    """filtro de magnitud Butterworth"""
    # corte = w en imagen de lado 1
    # 1 \over 1 + {D \over w}^{2n}
    magnitud = np.zeros((rows, cols))
    corte *= rows
    for k in range(rows):
        for l in range(cols):
            d2 = dist([k,l], [rows//2, cols//2])
            magnitud[k,l] = 1.0/(1 + (d2/corte)**order)
    return np.fft.ifftshift(magnitud)

def filterButterworthPasaAltos(rows, cols, corte, order):
    """filtro de magnitud Butterworth"""
    # corte = w en imagen de lado 1
    # 1 \over 1 + {D \over w}^{2n}
    magnitud = np.zeros((rows, cols))
    corte *= rows
    for k in range(rows):
        for l in range(cols):
            d2 = dist([k,l], [rows//2, cols//2])
            magnitud[k,l] = 1.0/(1 + (corte/d2)**order)
    return np.fft.ifftshift(magnitud)

def motionBlur(size, a, b):
    """Filtro de movimiento en direcciones a y b"""
    transformation = np.zeros(size)
    rows = size[0]
    cols = size[1]
    # fase exp(j\pi (ua + vb))
    # magnitud \frac{ \sin(\pi(ua+vb)) }{ \pi (ua+vb) }
    for k in range(rows):
        for l in range(cols):
            u = (l - cols/2)/cols
            v = (k - rows/2)/rows
            pi_v = ma.pi * (u*a + v*b)
            if pi_v:
                mag = np.sin(pi_v)/pi_v
            else:
                mag = 1  # lim{x->0} sin(x)/x
            transformation[k, l] = mag*np.exp(complex(0, 1)*pi_v)
    return np.fft.fftshift(transformation)

def filterHomomorfico(rows, cols, corte, gL, gH, order):
    """Filtro de magnitud homomorfico"""
    magnitud = np.zeros((rows, cols))
    corte *= rows
    for k in range(rows):
        for l in range(cols):
            d2 = dist([k,l], [rows//2, cols//2])
            magnitud[k,l] = (gH - gL)*(1 - np.exp(-order*(d2/(corte**2)))) + gL
    return np.fft.ifftshift(magnitud)

def imgCompare(img1,img2):
    plt.figure()
    plt.subplot(1,2,1)
    plt.imshow(img1, cmap='gray')
    plt.subplot(1,2,2)
    plt.imshow(img2, cmap='gray')
    plt.show()
