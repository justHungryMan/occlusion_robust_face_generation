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
from network import ResidualBlock, Generator, Discriminator, Conv2dSame


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
        gt_path = os.path.join(self.root_dir, "gt_resized")
        mask_deleted_path = os.path.join(self.root_dir, "mask_deleted")

        gt = Image.open(f'{gt_path}/{self.images[idx]}')
        mask_deleted = Image.open(mask_deleted_path + '/' + self.images[idx])  

        gt = self.transforms(gt)  
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


        return gt, mask_deleted
    
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

    landmark_pred = timm.create_model("tf_efficientnet_b1_ns", pretrained=False, num_classes=68 * 2)
    landmark_pred.conv_stem = Conv2dSame(4, 32, kernel_size=(3, 3), stride=(2, 2), bias=False)

    G = Generator(ResidualBlock, layer_count=7)
    D = Discriminator()

    G_optimizer = optim.AdamW(G.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    G_scheduler = optim.lr_scheduler.CosineAnnealingLR(G_optimizer, args.epochs, eta_min=0, last_epoch=-1)
    D_optimizer = optim.AdamW(D.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    D_scheduler = optim.lr_scheduler.CosineAnnealingLR(D_optimizer, args.epochs, eta_min=0, last_epoch=-1)

    if torch.cuda.is_available():
        G = nn.DataParallel(G)
        D = nn.DataParallel(D)
        landmark_pred = nn.DataParallel(landmark_pred)
        G = G.cuda()
        D = D.cuda()
        landmark_pred = landmark_pred.cuda()

    model_names = ["model"]
    _, model_list = load_ckp(args.landmark_model, model_names)
    landmark_pred.load_state_dict(model_list["model"])

    if args.checkpoint is not None:
        model_names = ["G", "D", "G_optimizer", "D_optimizer", "G_scheduler", "D_scheduler"]
        start_epoch, model_list = load_ckp(args.checkpoint, model_names)
        
        G.load_state_dict(model_list["G"])
        D.load_state_dict(model_list["D"])
        G_optimizer.load_state_dict(model_list["G_optimizer"])
        D_optimizer.load_state_dict(model_list["D_optimizer"])
        G_scheduler.load_state_dict(model_list["G_scheduler"])
        D_scheduler.load_state_dict(model_list["D_scheduler"])
    
    

    summary(G, input_size=(4, 256, 256))
    
    summary(D, input_size=(3, 256, 256))

    G.apply(weights_init_normal)
    D.apply(weights_init_normal)
    print('[Start] : Face reconstruction')

    logger = Logger(args.epochs, len(data_loader), image_step=10, env=args.result_name)

    for epoch in range(args.epochs):
        epoch = epoch + start_epoch + 1
        #print("Epoch[{epoch}] : Start".format(epoch=epoch))
        g_loss = 0
        for step, (gt, mask_deleted) in enumerate(data_loader):
            gt = to_variable(gt)
            mask_deleted = to_variable(mask_deleted)

            with torch.no_grad():
                landmark = landmark_pred(mask_deleted)

            batch_size = landmark.shape[0]
            aug_channel = torch.zeros((batch_size, 256, 256))

            for i in range(batch_size):
                for j in range(68):
                    x = int(landmark[i][j * 2])
                    y = int(landmark[i][j * 2 + 1])
                    if x >= 255:
                        x = 255
                    if y >= 255:
                        y = 255
                    if x < 0:
                        x = 0
                    if y < 0:
                        y = 0
                    aug_channel[i][x][y] = 1.
            mask_deleted[:, 3, :, :] = to_variable(aug_channel)

            fake_image = G(mask_deleted)
            real_image = gt

            D_optimizer.zero_grad()

            if args.loss == "ls":
                d_loss = patch_loss(criterion, D(real_image), True) + patch_loss(criterion, D(fake_image), False)
            elif args.loss == "wgan-gp":
                d_loss_real = D(real_image)
                d_loss_real = -torch.mean(d_loss_real)

                d_loss_fake = D(fake_image)
                d_loss_fake = torch.mean(d_loss_fake)

                d_loss_l1 = euclidean_l1(real_image, fake_image)
                grad_penalty = calculate_gradient_penalty(real_image, fake_image, D)
                
                Wasserstein_D = d_loss_real + d_loss_fake

                d_loss = d_loss_real + d_loss_fake + grad_penalty * args.lamda_gp + d_loss_l1 * args.lamda

            d_loss.backward()
            D_optimizer.step()
            D_scheduler.step()

            
            #if (step) % opt.n_critic == 0:
            if (step) % 1 == 0:
                G_optimizer.zero_grad()

                fake_image = G(mask_deleted)
                real_image = gt

                if args.loss == "ls":
                    g_loss = patch_loss(criterion, D(fake_image), True) + euclidean_l1(real_image, fake_image) * args.lamda
                elif args.loss == "wgan-gp":
                    g_loss_fake = -torch.mean(D(fake_image))
                    g_loss_rec = euclidean_l1(real_image, fake_image)

                    g_loss = g_loss_fake + args.lamda * g_loss_rec

                g_loss.backward()
                G_optimizer.step()
                G_scheduler.step()


            #if (step + 1 ) % opt.save_step == 0:
            if (step) % 2 == 0:
                #print("Epoch[{epoch}]| Step [{now}/{total}]| d_loss: {d_loss}, g_loss: {g_loss}".format(
                #    epoch=epoch, now=step + 1, total=len(data_loader), d_loss=d_loss, g_loss=g_loss,
                #    ))
                #batch_image = torch.cat((torch.cat((gt, masking_data), 3), torch.cat((gt_out, masking_out), 3)), 2)
                img_list = utils.make_grid([real_image[0], fake_image[0], mask_deleted[0][:-1]])
                utils.save_image(denorm(img_list), args.training_result + 'result_{result_name}_ep{epoch}_{step}.jpg'.format(result_name=args.result_name,epoch=epoch, step=(step + 1) * args.batch_size))
            # http://localhost:8097
            logger.log(
                losses={
                    'd_loss': d_loss,
                    'g_loss': g_loss,
                },
                images={
                    'real_image': real_image,
                    'fake_image': fake_image,
                    'masking_image': mask_deleted[:][:-1],
                },
            )
        if epoch % 10 == 0:
            torch.save({
                'epoch': epoch,
                'G': G.state_dict(),
                'D': D.state_dict(),
                'G_optimizer': G_optimizer.state_dict(),
                'D_optimizer': D_optimizer.state_dict(),
                'G_scheduler': G_scheduler.state_dict(),
                'D_scheduler': D_scheduler.state_dict(),
            }, args.save_model + 'model_{result_name}_ep{epoch}.ckp'.format(result_name=args.result_name, epoch=epoch))        

if __name__ == "__main__":
    torch.multiprocessing.set_sharing_strategy('file_system')
    parser = argparse.ArgumentParser(description='Pytorch Face Reconsturction')

    parser.add_argument('--dataset', required=True, type=str, help='gt_dataset path')
    parser.add_argument('--save_model', required=True, type=str, help='model save directory')
    parser.add_argument('--landmark_model', required=True, type=str, help='landmark predictor')
    parser.add_argument('--checkpoint', type=str, help='model checkpoint')
    parser.add_argument('--save_step', default=10, type=int, help='save step')
    parser.add_argument('--training_result', required=True, type=str, help='training_result')
    parser.add_argument('--result_name', default="landmark_pred", type=str, help='model saving name')
    parser.add_argument('--loss', default="ls", choices=['ls', 'wgan-gp'], type=str, help='model saving name')

    parser.add_argument('--batch_size', default=4, type=int, help='batch size')
    parser.add_argument('--epochs', default=10, type=int, help='epochs')
    parser.add_argument('--gpus', default="0", type=str, help='gpus')
    parser.add_argument('--lr', default=1e-3, type=float, help='learning rate')
    parser.add_argument('--weight_decay', default=1e-5, type=float, help='learning rate')
    parser.add_argument('--beta1', default=0.5, type=float, help='beta 1')
    parser.add_argument('--beta2', default=0.999, type=float, help='beta 2')    
    parser.add_argument('--lamda', default=10, type=int, help='lamda')
    parser.add_argument('--lamda_gp', default=10, type=int, help='lamda_gp')
    parser.add_argument('--n_critic', default=5, type=int, help='G update every n step')
    parser.add_argument('--num_workers', default=16, type=int, help='number of workers')

    args = parser.parse_args()
    main()