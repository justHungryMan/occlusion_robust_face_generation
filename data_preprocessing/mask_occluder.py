########################################################################################################################
#                                                                                                                      #
#                  Function: mask_occluder()                                                                           #
#                                                                                                                      #
#                  Description: ???                                                                                    #
#                                                                                                                      #
#                                                                                                                      #
#                  Date: 2019.04.02                                                                                    #
#                                                                                                                      #
#                  Author: Sidra Riaz (Python version by saleh@sogang.ac.kr)                                           #
#                                                                                                                      #
########################################################################################################################
# import libraries
import os
import cv2
import imutils
import glob
import math
#import image
from PIL import Image
import tifffile
import numpy as np
from random import random
import scipy.io as sio
import matplotlib.pyplot as mpl
import matplotlib.image as mplimg
from solve_foreground_background import solve_foreground_background

from find_min_max_ import find_min_max_
from after_maxima import after_maxima
from after_minima import after_minima
from faceCrop import faceCrop
from read_transparent_png import read_transparent_png

existed_file = os.listdir("1 Face Occlusion DB/5 results/")

def mask_occluder(inDir_Occluders, example_Img, coordinatesAll, fileName, output_dir):
    fileName_tiff_ = fileName[:-4].split("/")[-1] + ".tiff"
    if "1_" + fileName_tiff_ in existed_file:
        print("pass")
        return 0
    print(fileName)
    for i in range(1, 11): #Check this range later. Saleh

        occlusion_Type = 5
        mask_Type = i
        '''
        5: left_eye_y - 1
        6: left_eye_x - 0
        7: right_eye_y - 3
        8: right_eye_x - 2
        9: nose_y - 5
        10: nose_x - 4
        11: mouth_y
        12: mouth_x
        '''
        '''
        mask_ht1 = math.floor(coordinatesAll[5] - (coordinatesAll[0] - coordinatesAll[2]) / 6)
        if mask_ht1 < 0:
            mask_ht1 = 1
        mask_ht2 = math.floor(coordinatesAll[5] + (coordinatesAll[0] - coordinatesAll[2])*1.3)
        if mask_ht2 > example_Img.shape[0]:
            mask_ht2 = example_Img.shape[0]    
        '''
        #Jun
        mask_ht1 = math.floor(coordinatesAll[5] - (coordinatesAll[0] - coordinatesAll[2]) / (2 + (random() - 0.5)))
        if mask_ht1 < 0:
            mask_ht1 = 1

        
        mask_ht2 = math.floor(coordinatesAll[5] + (coordinatesAll[0] - coordinatesAll[2])*(1.3 + (random() - 0.5) / 2) - 10)
        if mask_ht2 > example_Img.shape[0]:
            mask_ht2 = example_Img.shape[0]
        #
        mask_length = abs(mask_ht2 - mask_ht1)
        
        mask_wd1 = round(coordinatesAll[2] - (coordinatesAll[0] - coordinatesAll[2]) / 1.7 + 10 * (random() - 0.5))
        if mask_wd1 < 0:
            crop_val1 = abs(mask_wd1)
        else:
            crop_val1 = 0

        mask_wd2 = round(coordinatesAll[0] + (coordinatesAll[0] - coordinatesAll[2]) / 1.7 - 10 * (random() - 0.5))
        if mask_wd2 > example_Img.shape[1]:
            crop_val2 = abs(example_Img.shape[1] - mask_wd2)
        else:
            crop_val2 = 0

        mask_width = abs(mask_wd2 - mask_wd1)

        file_mask_Img = './' + str(inDir_Occluders) + '/' + str(occlusion_Type) + '/' + str(mask_Type) + '.png'
        file_maskMask = './' + str(inDir_Occluders) + '/' + str(occlusion_Type) + '/' + str(mask_Type) + \
                        '-mask.jpg'

        mask_Img = read_transparent_png(file_mask_Img)
        maskMask = cv2.imread(file_maskMask, cv2.IMREAD_UNCHANGED)
        mask_resized = cv2.resize(np.float32(mask_Img), (np.int(mask_width), np.int(mask_length)))
        size_mask = mask_resized.shape

        mask_resized = after_minima(mask_resized, mask_resized.min())
        mask_resized = after_maxima(mask_resized, mask_resized.max())

        minima = maskMask[:, :, 0].min()
        maxima = maskMask[:, :, 0].max()
        maskMask_resized = cv2.resize(np.float32(maskMask[:, :, 0]), (np.int(mask_width), np.int(mask_length)))
        
        if random() > 0.5:
            mask_resized = cv2.flip(mask_resized, 1)
            maskMask_resized = cv2.flip(maskMask_resized, 1)
        # Rotate
        angle = (random()- 0.5) * 2 * 10 # -10 ~ 10
        mask_resized = imutils.rotate(mask_resized, angle)
        maskMask_resized = imutils.rotate(maskMask_resized, angle)
        
        for row in range(0, maskMask_resized.shape[0]):
            for col in range(0, maskMask_resized.shape[1]):
                maskMask_resized[row, col] = maskMask_resized[row, col] - minima

        for row in range(0, maskMask_resized.shape[0]):
            for col in range(0, maskMask_resized.shape[1]):
                maskMask_resized[row, col] = maskMask_resized[row, col] / maxima

        example_Img_occluded = np.float32(example_Img)
        example_Img_occluded = after_minima(example_Img_occluded, example_Img_occluded.min())
        example_Img_occluded = after_maxima(example_Img_occluded, example_Img_occluded.max())

        fg_, bg_ = solve_foreground_background(mask_resized, maskMask_resized)

        example_Img_occluded[np.int(mask_ht1) + 1: np.int(mask_ht2), np.int(mask_wd1) + \
                    np.int(crop_val1) + 1: np.int(mask_wd2) - np.int(crop_val2), 0] = \
        fg_[1: size_mask[0], np.int(crop_val1) + 1: size_mask[1] - np.int(crop_val2), 0] * \
        maskMask_resized[1: size_mask[0], np.int(crop_val1) + 1: size_mask[1] - np.int(crop_val2)] + \
        example_Img_occluded[np.int(mask_ht1) + 1: np.int(mask_ht2), np.int(mask_wd1) \
                    + 1 + np.int(crop_val1): np.int(mask_wd2) - np.int(crop_val2), 0] * \
        (1 - maskMask_resized[1:size_mask[0], np.int(crop_val1) + 1: size_mask[1] - np.int(crop_val2)])

        example_Img_occluded[np.int(mask_ht1) + 1: np.int(mask_ht2), np.int(mask_wd1) + \
                    np.int(crop_val1) + 1: np.int(mask_wd2) - np.int(crop_val2), 1] = \
        fg_[1: size_mask[0], np.int(crop_val1) + 1: size_mask[1] - np.int(crop_val2), 1] * \
        maskMask_resized[1: size_mask[0], np.int(crop_val1) + 1: size_mask[1] - np.int(crop_val2)] + \
        example_Img_occluded[np.int(mask_ht1) + 1: np.int(mask_ht2), np.int(mask_wd1) \
                    + 1 + np.int(crop_val1): np.int(mask_wd2) - np.int(crop_val2), 1] * \
        (1 - maskMask_resized[1:size_mask[0], np.int(crop_val1) + 1: size_mask[1] - np.int(crop_val2)])

        example_Img_occluded[np.int(mask_ht1) + 1: np.int(mask_ht2), np.int(mask_wd1) + \
                    np.int(crop_val1) + 1: np.int(mask_wd2) - np.int(crop_val2), 2] = \
        fg_[1: size_mask[0], np.int(crop_val1) + 1: size_mask[1] - np.int(crop_val2), 2] * \
        maskMask_resized[1: size_mask[0], np.int(crop_val1) + 1: size_mask[1] - np.int(crop_val2)] + \
        example_Img_occluded[np.int(mask_ht1) + 1: np.int(mask_ht2), np.int(mask_wd1) \
                    + 1 + np.int(crop_val1): np.int(mask_wd2) - np.int(crop_val2), 2] * \
        (1 - maskMask_resized[1:size_mask[0], np.int(crop_val1) + 1: size_mask[1] - np.int(crop_val2)])

       
        '''
        croppedImg = faceCrop(example_Img_occluded, coordinatesAll)

        # get current path
        current_path = os.getcwd()
        # get new path
        new_path = inDir_Occluders + '/' + str(occlusion_Type) + ' cropped/'
        # change path to new
        os.chdir(new_path)
        # write file
        file_name_ = str(i) + '_' + fileName_tiff_
        tifffile.imsave(file_name_, croppedImg, 'float32')
        # go back
        os.chdir(current_path)
        '''
        # get new path
        new_path = os.path.join(output_dir, 'masked_resized/')
        # change path to new
        file_name_ = str(i)  + '_' + fileName_tiff_
        tifffile.imsave(os.path.join(new_path, file_name_), example_Img_occluded, 'float32')
        

    return example_Img_occluded
