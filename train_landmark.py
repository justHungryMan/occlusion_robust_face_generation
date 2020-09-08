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



        return mask_deleted, gt_landmark
    
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


    opt = optim.AdamW(model.parameters(), lr=args.lr)
    

    if args.checkpoint is not None:
        model_names = ["model", "opt"]
        start_epoch, model_list = load_ckp(args.checkpoint, model_names)
        
        model.load_state_dict(model_list["model"])
        opt.load_state_dict(model_list["opt"])
    
    if torch.cuda.is_available():
        model = nn.DataParallel(model)
        model = model.cuda()

    summary(model, input_size=(4, 256, 256))

    #model.apply(weights_init_normal)
    print('[Start] : Face Landmark')

    logger = Logger(args.epochs, len(data_loader), image_step=10, env=args.result_name)

    for epoch in range(args.epochs):
        epoch = epoch + start_epoch + 1
        g_loss = 0
        for step, (mask_deleted, gt_landmark) in enumerate(data_loader):
            mask_deleted = to_variable(mask_deleted)
            gt_landmark = to_variable(gt_landmark)
            
            pred_landmark = model(mask_deleted)

            opt.zero_grad()

  
            loss = 0
            
            for i in range(68):
                loss += (pred_landmark[:, 2 * i] - gt_landmark[:, 2 * i]) ** 2 + (pred_landmark[:, 2 * i + 1] - gt_landmark[:, 2 * i + 1]) ** 2
            loss = loss.mean()

            loss.backward()
            opt.step()
            
            # http://localhost:8097
            logger.log(
                losses={
                    'loss': loss,
                },
                images={

                }
            )
        if epoch % 10 == 0:
            torch.save({
                'epoch': epoch,
                'model': model.state_dict(),
                'opt': opt.state_dict(),
            }, args.save_model + 'model_{result_name}_ep{epoch}.ckp'.format(result_name=args.result_name, epoch=epoch))        

if __name__ == "__main__":
    torch.multiprocessing.set_sharing_strategy('file_system')
    parser = argparse.ArgumentParser(description='Pytorch Face Reconsturction')

    parser.add_argument('--dataset', required=True, type=str, help='gt_dataset path')
    parser.add_argument('--save_model', required=True, type=str, help='model save directory')
    parser.add_argument('--checkpoint', type=str, help='model checkpoint')
    parser.add_argument('--save_step', default=10, type=int, help='save step')
    parser.add_argument('--result_name', default="landmark_pred", type=str, help='model saving name')
    parser.add_argument('--loss', default="ls", choices=['ls'], type=str, help='model saving name')

    parser.add_argument('--batch_size', default=4, type=int, help='batch size')
    parser.add_argument('--epochs', default=10, type=int, help='epochs')
    parser.add_argument('--gpus', default="0", type=str, help='gpus')
    parser.add_argument('--lr', default=0.0002, type=float, help='learning rate')
    parser.add_argument('--beta1', default=0.5, type=float, help='beta 1')
    parser.add_argument('--beta2', default=0.999, type=float, help='beta 2')    
    parser.add_argument('--lamda', default=10, type=int, help='lamda')
    parser.add_argument('--lamda_gp', default=10, type=int, help='lamda_gp')
    parser.add_argument('--n_critic', default=5, type=int, help='G update every n step')
    parser.add_argument('--num_workers', default=16, type=int, help='number of workers')

    args = parser.parse_args()
    main()