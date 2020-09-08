import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader

from torchvision import transforms, utils
import torchvision
from PIL import Image

import numpy
import argparse
import glob
import os
import datetime
import visdom
import json
from torchsummary import summary
from tqdm import tqdm
import timm

from utils import load_ckp, to_variable, denorm, Logger, weights_init_normal, calculate_gradient_penalty
from network import ResidualBlock, Generator, Discriminator, Conv2dSame


class image_preprocessing(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.transforms = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        self.images = []
        self.landmarks = []
        self.landmark_path = os.path.join(self.root_dir, "landmark.json") 

        with open(self.landmark_path) as json_file:
            json_data = json.load(json_file)

            for name in json_data:
                self.images.append(name)
                self.landmarks.append(json_data[name])

    def __getitem__(self, idx):
        gt_path = os.path.join(self.root_dir, "gt_resized")
        masked_path = os.path.join(self.root_dir, "mask_deleted")

        gt = Image.open(gt_path + '/' + self.images[idx])
        masking = Image.open(masked_path + '/' + self.images[idx])
        
        gt = self.transforms(gt)
        masking = self.transforms(masking)

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
            aug_channel[x][y] = 1
        aug_channel = aug_channel.unsqueeze(0)

        masking = torch.cat([masking, aug_channel], dim=0)

        return gt, masking
    
    def __len__(self):
        return len(self.images)

def main():
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
    start_epoch = 0

    pred_model = timm.create_model("tf_efficientnet_b1_ns", pretrained=False, num_classes=68 * 2)
    pred_model.conv_stem = Conv2dSame(4, 32, kernel_size=(3, 3), stride=(2, 2), bias=False)

    if torch.cuda.is_available():
        pred_model = nn.DataParallel(pred_model)
        pred_model = pred_model.cuda()
    model_names = ["model", "opt"]
    start_epoch, model_list = load_ckp(args.landmark_checkpoint, model_names)
    
    pred_model.load_state_dict(model_list["model"])

    dataset = image_preprocessing(args.dataset)
    data_loader = DataLoader(dataset, batch_size=args.batch_size,
                            shuffle=False, num_workers=args.num_workers)

    G = Generator(ResidualBlock, layer_count=7)

    if torch.cuda.is_available():
            G = nn.DataParallel(G)
            G = G.cuda()
    if args.checkpoint is not None:
        model_names = ["G"]
        start_epoch, model_list = load_ckp(args.checkpoint, model_names)
        
        G.load_state_dict(model_list["G"])

    G.eval()
    pred_model.eval()

    summary(G, input_size=(4, 256, 256))
    summary(pred_model, input_size=(4, 256, 256))

    print('[Start] : Infer Face')

    for step, (gt, masking_data) in tqdm(enumerate(data_loader)):
        gt = to_variable(gt)
        masking_data = to_variable(masking_data)
        
        with torch.no_grad():
            pred_landmark = pred_model(masking_data)
            batch_size = pred_landmark.shape[0]

            aug_channel = torch.zeros((batch_size, 256, 256))

            for i in range(batch_size):
                for j in range(68):
                    y = int(pred_landmark[i, 2 * j])
                    x = int(pred_landmark[i, 2 * j + 1])

                    if x >= 255:
                        x = 255
                    if y >= 255:
                        y = 255
                    if x < 0:
                        x = 0
                    if y < 0:
                        y = 0

                    aug_channel[i, x, y] = 1.

            aug_channel = to_variable(aug_channel)

            masking_data[:, 3, :, :] = aug_channel

            fake_image = G(masking_data)
            real_image = gt

        for i in range(real_image.shape[0]):
            img_list = utils.make_grid([real_image[i], fake_image[i], masking_data[i][:-1]])
            utils.save_image(denorm(img_list), args.result + 'result_{result_name}_{step}_mySynthesis.jpg'.format(result_name=args.result_name, step=step * args.batch_size + i))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Pytorch Face Reconsturction')

    parser.add_argument('--dataset', required=True, type=str, help='gt_dataset path')
    parser.add_argument('--landmark_checkpoint', required=True,type=str, help='landmark model checkpoint')
    parser.add_argument('--checkpoint', required=True,type=str, help='model checkpoint')
    parser.add_argument('--result_name', default="face_reconstruction", type=str, help='model saving name')
    parser.add_argument('--result', type=str, help='result path')

    parser.add_argument('--batch_size', default=4, type=int, help='batch size')
    parser.add_argument('--gpus', default="0", type=str, help='batch size')
    parser.add_argument('--num_workers', default=0, type=int, help='number of workers')

    args = parser.parse_args()
    main()
