import dlib
import cv2
import numpy as np
import os
from glob import glob
import argparse
import json
from collections import OrderedDict
from skimage import io
from shutil import copyfile
from tqdm import tqdm

def main(opt):
    file_list = os.listdir(opt.dataset)

    for fileName in tqdm(file_list):
        for i in range(1, 11):
            copyfile(os.path.join(opt.dataset, fileName), os.path.join(opt.dataset, str(i) + '_' + fileName))
            

if __name__ == "__main__":
        parser = argparse.ArgumentParser(description='Occluded face generation')

        parser.add_argument('--dataset', required=True, type=str, help='original dataset path')

        opt = parser.parse_args()
        main(opt)