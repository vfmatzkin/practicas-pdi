# -*- coding: utf-8 -*-

import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
import funciones as func
import pdifun as fun
import math


def parcial2018():
    img = cv.imread('archivo.jpg')
    img_salida = img.copy()
    img_g = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    img_hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    rows, cols, c = img.shape

    func.graficar(img[:,:,::-1], 255, 0, 'gray', 'Entrada')

    func.graficar(img[:,:,::0], 255, 0, 'gray', 'R')
    func.graficar(img[:,:,::1], 255, 0, 'gray', 'G')
    func.graficar(img[:,:,::2], 255, 0, 'gray', 'B')

    func.histograma(img[:,:,::0], 'Histograma de R')
    func.histograma(img[:,:,::1], 'Histograma de G')
    func.histograma(img[:,:,::2], 'Histograma de B')

    func.graficar(img_hsv[:,:,::0], 255, 0, 'gray', 'H')
    func.graficar(img_hsv[:,:,::1], 255, 0, 'gray', 'S')
    func.graficar(img_hsv[:,:,::2], 255, 0, 'gray', 'V')

    func.histograma(img_hsv[:,:,::0], 'Histograma de H')
    func.histograma(img_hsv[:,:,::1], 'Histograma de S')
    func.histograma(img_hsv[:,:,::2], 'Histograma de V')

    # ret, th_spec = cv.threshold(img_g, 235, 255, cv.THRESH_BINARY)

    # lines, points = func.hough(th_spec, n = 1, plotimg = img_salida, color = (0, 0, 255))

    # mask = func.segmentarColorHS(img_hsv, img_hsv, 5)  # Le paso solo la cancha (no la tribuna)

    # cuad = cv.getStructuringElement(cv.MORPH_RECT, (3, 3))
    # erod1 = cv.erode(mask, cuad, iterations = 9)
    # dilat1 = cv.dilate(erod1, cuad, iterations = 9)

    # median1 = cv.medianBlur(dilat1, 5)
    # blur1 = cv.GaussianBlur(median1, (9, 9), 0)

    func.graficar(img_salida[:,:,::-1], 255, 0, 'gray', 'Salida')

if __name__ == '__main__':
    parcial2018()
    plt.show()