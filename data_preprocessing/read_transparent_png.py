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


def read_transparent_png(filename):
    image_4channel = cv2.imread(filename, cv2.IMREAD_UNCHANGED)
    alpha_channel = image_4channel[:, :, 3]
    rgb_channels = image_4channel[:, :, :3]

    # White Background Image
    white_background_image = np.ones_like(rgb_channels, dtype=np.uint8) * 0

    # Alpha factor
    alpha_factor = alpha_channel[:, :, np.newaxis].astype(np.float32) / 255.0
    alpha_factor = np.concatenate((alpha_factor, alpha_factor, alpha_factor), axis=2)

    # Transparent Image Rendered on White Background
    base = rgb_channels.astype(np.float32) * alpha_factor
    white = white_background_image.astype(np.float32) * (1 - alpha_factor)
    final_image = base + white

    return cv2.cvtColor(final_image.astype(np.uint8), cv2.COLOR_BGR2RGB)
