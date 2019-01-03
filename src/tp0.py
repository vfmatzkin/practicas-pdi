# -*- coding: utf-8 -*-

import numpy as np
import cv2 as cv


print(cv.__version__)

img1 = cv.imread("camino.tif")  # RGB

# Las funciones reciben banderas (numeros enteros)
img2 = cv.imread("camino.tif", cv.IMREAD_GRAYSCALE)  # escala de grises
cv.imshow("Titulo 1", img1)
cv.imshow("Titulo 2", img2)
cv.waitKey(0)
cv.destroyAllWindows()

# Usando la misma ventana
cv.namedWindow("imagen")
cv.imshow("imagen", img1)
cv.waitKey(0)
cv.imshow("imagen", img2)
cv.waitKey(0)
cv.destroyAllWindows()

'''Crear una imagen vacı́a (en este caso una máscara). Las imágenes se representan
en matrices de Numpy de forma [alto,ancho,Ncanales], con un tipo de dato determinado
(flotante,entero,booleano, etc.), por tanto se pueden utilizar todas las propiedades
de Numpy.'''

H, W, C = img1.shape
imtype = img1.dtype
print("Imagen de tipo %s, alto=%d, ancho=%d, ncanales=%d" %(imtype, H, W, C))
mask=np.zeros((H, W, C), dtype=imtype)  # dtype: 'int8', bool, float, ...

# O bien:
mask = img1.copy()
mask[:] = 0

'''Las imágenes son objetos y para hacer copias se debe tener en cuenta que el operador
de asignación ("=") copia también el puntero al objeto.'''
mask[int(H/3):int(2*H/3), int(W/3):int(2*W/3), :] = 255  # Máscara rectangular, con valores [0,255]

# operación AND para cada pixel
img3 = cv.bitwise_and(img1, mask)

# Guardar la nueva imagen
cv.imwrite("nueva.jpg", img3)

# Capturar imágenes desde un stream de video.
# origen='archivo'
origen = 'webcam'
if origen == "archivo":
    cam=cv.VideoCapture("video.mp4")

# Video: a partir de camara habilitada por defecto
if origen == 'webcam':
    cam=cv.VideoCapture(0)

cv.namedWindow("Video")

# capturar 10 frames
for k in range(10):
    isok, frame = cam.read()
    if isok:
        cv.imshow("Video", frame)
        cv.waitKey(0)

cam.release()  # liberar stream
