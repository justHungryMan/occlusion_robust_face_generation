# import libraries
import os
import cv2
import glob
import math
#import image
from PIL import Image
#import tifffile
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as mpl
import matplotlib.image as mplimg
from solve_foreground_background import solve_foreground_background

def find_min_max_(src, maxima, minima):

    src.shape
    src.shape[2]

    if src.shape[2] == 3:
        bgr = src[:, :, :3]

        for row in range(0, src.shape[0]):
            for col in range(0, src.shape[1]):
                if bgr[row, col, 0] < minima:
                    minima = bgr[row, col, 0]
                if bgr[row, col, 0] > maxima:
                    maxima = bgr[row, col, 0]
        for row in range(0, src.shape[0]):
            for col in range(0, src.shape[1]):
                if bgr[row, col, 1] < minima:
                    minima = bgr[row, col, 1]
                if bgr[row, col, 1] > maxima:
                    maxima = bgr[row, col, 1]
        for row in range(0, src.shape[0]):
            for col in range(0, src.shape[1]):
                if bgr[row, col, 2] < minima:
                    minima = bgr[row, col, 2]
                if bgr[row, col, 2] > maxima:
                    maxima = bgr[row, col, 2]

    else:
        bgr = src[:, :, :3]
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        alpha = src[:, :, 3]

        for row in range(0, src.shape[0]):
            for col in range(0, src.shape[1]):
                if alpha[row, col] != 0:
                    #print(bgr[row, col, 0])
                    if bgr[row, col, 0] < minima:
                        minima = bgr[row, col, 0]
                    if bgr[row, col, 0] > maxima:
                        maxima = bgr[row, col, 0]
        for row in range(0, src.shape[0]):
            for col in range(0, src.shape[1]):
                if alpha[row, col] != 0:
                    #print(bgr[row, col, 1])
                    if bgr[row, col, 1] < minima:
                        minima = bgr[row, col, 1]
                    if bgr[row, col, 1] > maxima:
                        maxima = bgr[row, col, 1]
        for row in range(0, src.shape[0]):
            for col in range(0, src.shape[1]):
                if alpha[row, col] != 0:
                    #print(bgr[row, col, 2])
                    if bgr[row, col, 2] < minima:
                        minima = bgr[row, col, 2]
                    if bgr[row, col, 2] > maxima:
                        maxima = bgr[row, col, 2]

    #print(maxima)
    #print(minima)

    return [maxima , minima]