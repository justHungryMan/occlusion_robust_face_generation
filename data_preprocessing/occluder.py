import cv2
import numpy as np
import os
import math
from glob import glob
import argparse
import json
from collections import OrderedDict
import face_alignment
from skimage import io
import matplotlib.pyplot as mpl
from random import random
from tqdm import tqdm
import imutils
import tifffile

from solve_foreground_background import solve_foreground_background

from find_min_max_ import find_min_max_
from after_maxima import after_maxima
from after_minima import after_minima
from read_transparent_png import read_transparent_png

def main(opt):
    landmark_path = os.path.join(opt.output, "landmark.json")
    inDir_Occluders = './Face_Occlusion_DB'
    with open(landmark_path) as json_file:
        json_data = json.load(json_file)
    
    for fileName in tqdm(json_data.keys()):
        landmarks = json_data[fileName]
        right_eye_x = (landmarks[38 - 1][0] + landmarks[39 - 1][0]) / 2
        right_eye_y = (landmarks[38 - 1][1] + landmarks[42 - 1][1]) / 2

        left_eye_x = (landmarks[44 - 1][0] + landmarks[45 - 1][0]) / 2
        left_eye_y = (landmarks[44 - 1][1] + landmarks[48 - 1][1]) / 2

        nose_x = (landmarks[33 - 1][0] + landmarks[35 - 1][0]) / 2
        nose_y = (landmarks[31 - 1][1] + landmarks[34 - 1][1]) / 2

        mouth_x = landmarks[67 - 1][0]
        mouth_y = (landmarks[67 - 1][1] + landmarks[63 - 1][1]) / 2
        


        coordinatesAll = np.array([float(left_eye_x), float(left_eye_y), float(right_eye_x), \
                    float(right_eye_y), float(nose_x), float(nose_y)])
        Img = mpl.imread(os.path.join(opt.dataset, "gt_resized", fileName))
        
        mask_occluder(inDir_Occluders, Img, coordinatesAll, fileName, opt.output)

def mask_occluder(inDir_Occluders, example_Img, coordinatesAll, fileName, output_dir):
    output_path = os.path.join(output_dir, 'masked_resized')
    if not os.path.exists(output_path):
            os.makedirs(output_path)

    fileName_tiff_ = fileName[:-4].split("/")[-1] + ".tiff"
    '''
    if "1_" + fileName_tiff_ in existed_file:
        print("pass")
        return 0
    print(fileName)
    '''
    for i in range(1, 11): #Check this range later. Saleh

        occlusion_Type = 5
        mask_Type = i

        #Jun
        mask_ht1 = math.floor(coordinatesAll[5] - (coordinatesAll[0] - coordinatesAll[2]) / (2 + (random() - 0.5)))
        if mask_ht1 < 0:
            mask_ht1 = 1

        
        mask_ht2 = math.floor(coordinatesAll[5] + (coordinatesAll[0] - coordinatesAll[2])*(1.3 + (random() - 0.5) / 2) - 20)
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
        angle = (random()- 0.5) * 2 * 15 # -15 ~ 15
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


        
        file_name_ = str(i)  + '_' + fileName_tiff_
        tifffile.imsave(os.path.join(output_path, file_name_), example_Img_occluded, 'float32')
        

    return example_Img_occluded



if __name__ == "__main__":
        parser = argparse.ArgumentParser(description='Occluded face generation')

        parser.add_argument('--dataset', required=True, type=str, help='original dataset path')
        parser.add_argument('--output', required=True, type=str, help='output dataset path')

        opt = parser.parse_args()
        main(opt)