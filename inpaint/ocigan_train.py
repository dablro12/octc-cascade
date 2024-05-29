#!/usr/bin/env python

import os
import numpy as np
import matplotlib.pyplot as plt 
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils.dataset import CustomDataset
from model.aotgan import InpaintGenerator, Discriminator
from loss.loss import L1, Perceptual, Style, smgan
import wandb
from utils.arg import save_args
# Set environment variable for compatibility issues
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

# Disable unnecessary precision to speed up computations
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
torch.autograd.set_detect_anomaly(True)

class Train(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.setup_device(args)
        self.setup_datasets(args)
        self.setup_wandb(args)
        self.initialize_models(args)
        self.setup_train(args)
        self.setup_paths(args)
        

        self.best_valid_loss = np.inf
        self.epochs_no_improve = 0
        self.n_epochs_stop = 10  # Number of epochs to wait before stopping

    def setup_device(self, args):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"CUDA Status : {self.device.type}")

    def setup_datasets(self, args):
        transform = {
            'train': transforms.Compose([
                transforms.Grayscale(num_output_channels=3),
                transforms.GaussianBlur(3, sigma=(0.1, 2.0)),
                transforms.RandomAdjustSharpness(0.5),
                transforms.RandomAutocontrast(0.5),
                # transforms.ColorJitter(0.05, 0.05, 0.05, 0.05),
                transforms.RandomRotation(degrees=(-15,15)),
                transforms.ToTensor()
            ]),
            'valid': transforms.Compose([
                transforms.Grayscale(num_output_channels=3),
                transforms.ToTensor(),
            ])
        }
        mask_transform = {
            'train': transforms.Compose([
                transforms.Grayscale(num_output_channels=3),
                transforms.RandomAdjustSharpness(0.5),
                transforms.RandomRotation(degrees=(-5,5)),
                transforms.ToTensor()
            ]),
            'valid': transforms.Compose([
                transforms.Grayscale(num_output_channels=3),
                transforms.ToTensor(),
            ])
        }
        img_dirs = {'train': '/mnt/HDD/octc/mask_abstract/train', 'valid': '/mnt/HDD/octc/mask_abstract/test'}
        mask_dir = '/mnt/HDD/octc/mask_abstract/mask'

        self.train_loader = DataLoader(
            CustomDataset(img_dirs['train'], mask_dir, transform['train'], mask_transform = mask_transform['train'], testing = False, mask_shuffle = True),
            batch_size=args.ts_batch_size, shuffle=True
        )
        self.valid_loader = DataLoader(
            CustomDataset(img_dirs['valid'], mask_dir, transform['valid'], mask_transform = mask_transform['valid'], testing = True, mask_shuffle = False), 
            batch_size=args.vs_batch_size, shuffle=True
        )

    def setup_wandb(self, args):
        self.w = args.wandb.strip().lower()
        if args.wandb.strip().lower() == "yes":
            wandb.init(project='PCOS-v2', entity='dablro1232', notes='baseline', config=args.__dict__)
            wandb.run.name = args.model + f'_{args.version}_{args.training_date}'
            self.run_name = args.model + f'_{args.version}_{args.training_date}'
        else:
            self.run_name = args.model + '_debug'

    def initialize_models(self, args):
        self.netG = InpaintGenerator().to(self.device)
        self.netD = Discriminator().to(self.device)
        self.setup_optimizers(args)
    
    def setup_train(self, args):
        self.epochs = args.epochs
        self.L1 = L1()
        self.Per = Perceptual()
        self.Style = Style()
        self.Gan = smgan()
        

    def setup_optimizers(self, args):
        self.optimG = optim.Adam(self.netG.parameters(), lr=args.learning_rate, betas=(0, 0.9))
        self.optimD = optim.Adam(self.netD.parameters(), lr=args.learning_rate, betas=(0, 0.9))

        if args.pretrain == 'yes':
            checkpoint_paths = {key: f"/mnt/HDD/oci_models/aotgan/{args.pretrained_model}/{key}0000000.pt" for key in ['G', 'D', 'O']}
            self.load_checkpoints(checkpoint_paths)
            print('\033[41m',"#"*30, ' | ', 'Pretrained Setting Complete !!', '\033[0m')

    def load_checkpoints(self, paths):
        self.G_checkpoint = torch.load(paths['G'], map_location=self.device)
        self.netG.load_state_dict(self.G_checkpoint)
        self.D_checkpoint = torch.load(paths['D'], map_location=self.device)
        self.netD.load_state_dict(self.D_checkpoint)
        self.O_checkpoint = torch.load(paths['O'], map_location=self.device)
        self.optimG.load_state_dict(self.O_checkpoint['optimG']) #Only use pretrained G optimizer checkpoint 
        # self.optimD.load_state_dict(self.O_checkpoint['optimD'])

    def setup_paths(self, args):
        self.save_path = os.path.join(args.save_path, f"{self.run_name}")
        os.makedirs(self.save_path, exist_ok=True)
        save_args(f"{self.save_path}/{self.run_name}.json")

    def fit(self):
        for epoch in tqdm(range(1, self.epochs+1)):
            train_g_loss, train_d_loss, valid_g_loss, valid_d_loss = 0., 0., 0., 0.

            # Training phase
            for images, masks in self.train_loader:
                images, input_images, masks = self.prepare_images(images, masks)

                pred_images = self.netG(input_images, masks) #Need to 3+1 channel 
                comp_images = self.compute_composite_images(images, pred_images, masks)

                # Loss computation
                g_loss, d_loss = self.compute_losses(images, pred_images, comp_images, masks)

                # Backpropagation
                self.optimG.zero_grad()
                self.optimD.zero_grad()
                g_loss.backward()
                d_loss.backward()
                self.optimG.step()
                self.optimD.step()

                train_g_loss += g_loss.item()
                train_d_loss += d_loss.item()

            # Validation phase
            with torch.no_grad():
                self.netG.eval()
                self.netD.eval()
                for (images, masks, paths) in self.valid_loader:
                    images, input_images, masks = self.prepare_images(images, masks)

                    pred_images = self.netG(input_images, masks)
                    comp_images = self.compute_composite_images(images, pred_images, masks)

                    # Loss computation
                    v_g_loss, v_d_loss = self.compute_losses(images, pred_images, comp_images, masks)

                    valid_g_loss += v_g_loss.item()
                    valid_d_loss += v_d_loss.item()
            self.log_metrics(epoch, train_g_loss, train_d_loss, valid_g_loss, valid_d_loss)
            self.visualize(epoch = epoch, image = images[0,0], mask = masks[0,0], input_image = input_images[0,0], comp_image= comp_images[0,0])
            
            # Early Stopping Check
            if valid_g_loss < self.best_valid_loss:
                self.best_valid_loss = valid_g_loss
                self.epochs_no_improve = 0
                self.save_model(epoch, valid_g_loss)  # Save the best model
            else:
                self.epochs_no_improve += 1

            if self.epochs_no_improve >= self.n_epochs_stop:
                print(f"\033[41m Early stopping at epoch {epoch}. Best valid loss: {self.best_valid_loss}. \033[0m")
                break

        if self.w == "yes":
            wandb.finish()
        print("Training Complete.")

    def prepare_images(self, images, masks):
        input_images = images.clone()
        input_images[masks != 0] = masks[masks != 0]
        return images.to(self.device), input_images.to(self.device), masks[:,0,:,:].unsqueeze(1).to(self.device)

    def compute_composite_images(self, images, pred_images, masks):
        comp_images = images.clone()
        comp_images[masks.repeat(1,3,1,1) != 0] = pred_images[masks.repeat(1,3,1,1) != 0]
        return comp_images

    def compute_losses(self, real_images, pred_images, comp_images, masks):
        l1 = self.L1(pred_images, real_images)
        per = self.Per(pred_images, real_images)
        style = self.Style(pred_images, comp_images)
        pred_mask, (d_loss, adv_loss) = self.Gan(self.netD, comp_images, real_images, masks)
        g_loss = adv_loss * 0.01 + l1 * 1 + per * 0.1 + style * 100
        return g_loss, d_loss

    def log_metrics(self, epoch, train_g_loss, train_d_loss, valid_g_loss, valid_d_loss):
        if self.w == 'yes':
            wandb.log({
                "train_G_loss": train_g_loss,
                "train_D_loss": train_d_loss,
                "valid_G_loss": valid_g_loss,
                "valid_D_loss": valid_d_loss
            }, step = epoch)

    def save_model(self, epoch, valid_g_loss):
        current_path = os.path.join(self.save_path, f"model_{epoch}.pt")
        torch.save({
            "epoch": epoch,
            "netG_state_dict": self.netG.state_dict(),
            "netD_state_dict": self.netD.state_dict(),
            "optimG_state_dict": self.optimG.state_dict(),
            "optimD_state_dict": self.optimD.state_dict(),
            "valid_g_loss": valid_g_loss
        }, current_path)
        print(f"Saved model at {current_path}")

    def visualize(self, epoch, image, mask, input_image, comp_image):
        current_path = os.path.join(self.save_path, f"result_{epoch}.png")

        vis_imgs = [image, mask, input_image, comp_image]
        vis_labels = ['GT', 'Mask', 'Input', 'Result']
        plt.figure(figsize = (12, 8))
        for index, plot_img in enumerate(vis_imgs):
            plt.subplot(2,2,index+1)
            plt.imshow(plot_img.cpu().detach().numpy(), cmap = 'gray')
            plt.title(vis_labels[index])
            plt.axis('off')
        plt.tight_layout()
        plt.savefig(current_path)
        plt.close()
        
        print(f"Visualize save : {epoch}/{self.epochs}")
        