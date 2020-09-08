import dlib
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
        detector = dlib.get_frontal_face_detector()

        image_dir = opt.dataset
        output_dir = opt.output
        output_resized_dir = os.path.join(output_dir, "gt_resized")

        if not os.path.exists(output_resized_dir):
            os.makedirs(output_resized_dir)

        paths = glob(image_dir + "/*.jpg")
        file_data = OrderedDict()
        temp = []
        for path in tqdm(paths):
            image_name = path.split('/')[-1]
            print(image_name)
            img = cv2.imread(path, cv2.IMREAD_COLOR)
            
            height, width, channels = img.shape
            
            # detect
            rect = detector(img, 1)
            if len(rect) == 0:
                rect = temp
                continue
            if rect is not None:
                temp = rect
                rect = rect[0]
            correction = 20
            top = rect.top() - correction if rect.top() - correction >= 0 else 0 
            bottom = rect.bottom() + correction if rect.bottom() + correction <= height else height
            left = rect.left() - correction if rect.left() - correction >= 0 else 0
            right = rect.right() + correction if rect.right() + correction <= width else width

            crop = img[top:bottom, left:right]

            crop_resized = cv2.resize(crop, (256, 256), interpolation=cv2.INTER_AREA)
            cv2.imwrite(os.path.join(output_resized_dir, image_name), crop_resized)
            
            # landmark
            input_image = io.imread(os.path.join(output_resized_dir, image_name))

            preds = fa.get_landmarks(input_image)
            if preds is None:
                continue
            dots = []
            for j in range(68):
                dots.append([int(preds[0][j][0]), int(preds[0][j][1])])
            file_data[image_name] = dots
        
        with open(os.path.join(output_dir, 'landmark.json'), 'w', encoding="utf-8") as make_file:
            json.dump(file_data, make_file, ensure_ascii=False, indent='\t')
            

if __name__ == "__main__":
        parser = argparse.ArgumentParser(description='Occluded face generation')

        parser.add_argument('--dataset', required=True, type=str, help='original dataset path')
        parser.add_argument('--output', required=True, type=str, help='output dataset path')

        opt = parser.parse_args()
        main(opt)