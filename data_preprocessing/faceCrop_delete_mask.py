import dlib
import math
import cv2
import numpy as np
import os
from glob import glob
import argparse
import json
from collections import OrderedDict
import face_alignment
from skimage import io

from tqdm import tqdm

def main(opt):
        fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, flip_input=False)

        image_dir = opt.dataset
        output_dir = opt.output
        output_resized_dir = os.path.join(output_dir, "mask_deleted")

        if not os.path.exists(output_resized_dir):
            os.makedirs(output_resized_dir)

        paths = glob(image_dir + "/*.jpg")
        file_data = OrderedDict()
        temp = []
        for path in tqdm(paths):
            image_name = path.split('/')[-1]
            img = cv2.imread(path, cv2.IMREAD_COLOR)
            
            height, width, channels = img.shape
          
            # landmark
            input_image = io.imread(os.path.join(image_dir, image_name))

            preds = fa.get_landmarks(input_image)
            if preds is None:
                continue
            dots = []

            possible_landmark = [0, 1, 2, 3, 27, 28, 16, 15, 14, 13, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47]
            
            for j in range(68):
                if j not in possible_landmark:
                    continue
                x = int(preds[0][j][0])
                y = int(preds[0][j][1])
                dots.append([x, y])

            left_x = int(preds[0][2][0])
            left_y = int(preds[0][28][1] + 10)
            right_x = int(preds[0][14][0])
            right_y = int(left_y + abs(right_x - left_x))

            img[left_y:right_y, left_x:right_x, :] = np.random.normal(loc=0, scale=1, size=(img[left_y:right_y, left_x:right_x, :].shape))
            #img = cv2.rectangle(img, (left_x, left_y), (right_x, right_y), (0,0,0), -1)

            cv2.imwrite(os.path.join(output_resized_dir, image_name), img)

            file_data[image_name] = dots
        
        with open(os.path.join(output_dir, 'landmark.json'), 'w', encoding="utf-8") as make_file:
            json.dump(file_data, make_file, ensure_ascii=False, indent='\t')
            

if __name__ == "__main__":
        parser = argparse.ArgumentParser(description='Occluded face generation')

        parser.add_argument('--dataset', required=True, type=str, help='original dataset path')
        parser.add_argument('--output', required=True, type=str, help='output dataset path')

        opt = parser.parse_args()
        main(opt)