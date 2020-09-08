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

def after_maxima(src, maxima):

    if src.shape[2] == 3:
        dest = np.float32(src)
        #dest = src.copy()

        for row in range(0, src.shape[0]):
            for col in range(0, src.shape[1]):
                dest[row, col, 0] = dest[row, col, 0] / maxima
        for row in range(0, src.shape[0]):
            for col in range(0, src.shape[1]):
                dest[row, col, 1] = dest[row, col, 1] / maxima
        for row in range(0, src.shape[0]):
            for col in range(0, src.shape[1]):
                dest[row, col, 2] = dest[row, col, 2] / maxima

    else:
        #dest = np.float32(src)
        dest = src.copy()
        alpha = src[:, :, 3]

        for row in range(0, src.shape[0]):
            for col in range(0, src.shape[1]):
                if alpha[row, col] != 0:
                    dest[row, col, 0] = src[row, col, 0] / maxima
                else:
                    dest[row, col, 0] = 0
        for row in range(0, src.shape[0]):
            for col in range(0, src.shape[1]):
                if alpha[row, col] != 0:
                    if alpha[row, col] != 0:
                        dest[row, col, 1] = src[row, col, 1] / maxima
                    else:
                        dest[row, col, 1] = 0
        for row in range(0, src.shape[0]):
            for col in range(0, src.shape[1]):
                if alpha[row, col] != 0:
                    if alpha[row, col] != 0:
                        dest[row, col, 2] = src[row, col, 2] / maxima
                    else:
                        dest[row, col, 2] = 0

    return dest