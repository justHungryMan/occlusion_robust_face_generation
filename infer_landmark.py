import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader

from torchvision import transforms, utils
import torchvision
from PIL import Image
import cv2
from tqdm import tqdm

import numpy
import argparse
import glob
import os
import datetime
import visdom
import json
import timm
from torchsummary import summary

from utils import load_ckp, to_variable, denorm, Logger, weights_init_normal, calculate_gradient_penalty
from network import Conv2dSame


class image_preprocessing(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.transforms = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        self.resize = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
        ])
        self.images = []
        self.landmarks = []
        self.gt_landmarks = []
        self.landmark_path = os.path.join(self.root_dir, "landmark.json") 
        self.gt_landmark_path = os.path.join(self.root_dir, "gt_landmark.json") 

        with open(self.landmark_path) as json_file:
            with open(self.gt_landmark_path) as json_gt_file:
                json_gt_data = json.load(json_gt_file)
                json_data = json.load(json_file)
                for name in json_gt_data:
                    for i in range(1, 11):
                        if str(i) + '_' + name not in json_data:
                            continue
                        self.gt_landmarks.append(json_gt_data[name])
                        self.images.append(str(i) + '_' + name)
                        self.landmarks.append(json_data[str(i) + '_' + name])
    def __getitem__(self, idx):
        mask_deleted_path = os.path.join(self.root_dir, "mask_deleted")

        mask_deleted = Image.open(mask_deleted_path + '/' + self.images[idx])    
        mask_deleted = self.transforms(mask_deleted)

        aug_channel = torch.zeros((256, 256))
        landmark = self.landmarks[idx]
        
        possible_landmark = [0, 1, 2, 3, 27, 28, 16, 15, 14, 13, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47]

        for (step, point) in enumerate(landmark):
            if step not in possible_landmark:
                continue
            y = point[0]
            x = point[1]

            if x >= 255:
                x = 255
            if y >= 255:
                y = 255
            if x < 0:
                x = 0
            if y < 0:
                y = 0
            aug_channel[int(x)][int(y)] = 1
        aug_channel = aug_channel.unsqueeze(0)

        mask_deleted = torch.cat([mask_deleted, aug_channel], dim=0)

        gt_landmark = torch.zeros((68 * 2))
        for i in range(68):
            gt_landmark[i * 2] = self.gt_landmarks[idx][i][0]
            gt_landmark[i * 2 + 1] = self.gt_landmarks[idx][i][1]



        return mask_deleted, gt_landmark, f'{os.path.join(self.root_dir, "gt_resized")}/{self.images[idx]}'
    
    def __len__(self):
        return len(self.images)

def patch_loss(criterion, input, TF):
    if TF is True:
        comparison = torch.ones_like(input)
    else:
        comparison = torch.zeros_like(input)
    return criterion(input, comparison)

def least_squares(input, comparision):
    return 0.5 * torch.mean((input - comparision) ** 2)

def main():
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
    start_epoch = 0
    dataset = image_preprocessing(args.dataset)
    data_loader = DataLoader(dataset, batch_size=args.batch_size,
                            shuffle=True, num_workers=args.num_workers)

    criterion = least_squares
    euclidean_l1 = nn.L1Loss()

    model = timm.create_model("tf_efficientnet_b1_ns", pretrained=False, num_classes=68 * 2)
    model.conv_stem = Conv2dSame(4, 32, kernel_size=(3, 3), stride=(2, 2), bias=False)

    if torch.cuda.is_available():
        model = nn.DataParallel(model)
        model = model.cuda()

    if args.checkpoint is not None:
        model_names = ["model", "opt"]
        start_epoch, model_list = load_ckp(args.checkpoint, model_names)
        
        model.load_state_dict(model_list["model"])
    
    

    summary(model, input_size=(4, 256, 256))

    #model.apply(weights_init_normal)
    print('[Start] : Face Landmark')
    count = 0
    for step, (mask_deleted, gt_landmark, mask_deleted_path) in tqdm(enumerate(data_loader)):
        mask_deleted = to_variable(mask_deleted)
        gt_landmark = to_variable(gt_landmark)
        
        pred_landmark = model(mask_deleted)

        batch_size = pred_landmark.shape[0]

        for i in range(batch_size):
            mask_deleted_image = cv2.imread(mask_deleted_path[i], cv2.IMREAD_COLOR)
            for j in range(68):
                x = int(pred_landmark[i, 2 * j])
                y = int(pred_landmark[i, 2 * j + 1])
                mask_deleted_image = cv2.circle(mask_deleted_image, (x, y), 1, (0, 0, 255), -1)

            cv2.imwrite(os.path.join(args.result, args.result_name) + f'{count}.jpg', mask_deleted_image)
            count += 1



if __name__ == "__main__":
    torch.multiprocessing.set_sharing_strategy('file_system')
    parser = argparse.ArgumentParser(description='Pytorch Face Reconsturction')

    parser.add_argument('--dataset', required=True, type=str, help='gt_dataset path')
    parser.add_argument('--checkpoint', type=str, help='model checkpoint')
    parser.add_argument('--result_name', default="landmark_pred", type=str, help='model saving name')
    parser.add_argument('--result', type=str, help='result path')

    parser.add_argument('--batch_size', default=128, type=int, help='batch size')
    parser.add_argument('--gpus', default="0", type=str, help='batch size')
    parser.add_argument('--num_workers', default=16, type=int, help='number of workers')

    args = parser.parse_args()
    main()