#!/usr/bin/env python
import sys 
sys.path.append('../')
import os

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchvision import transforms 
from torch.utils.data import DataLoader

import numpy as np

import matplotlib.pyplot as plt 

from tqdm import tqdm

import wandb

from utils.dataset import Segmentation_CustomDataset as CustomDataset
from utils.metrics import calculate_metrics
from utils.__init__ import *
from utils.arg import save_args
from utils import * 

from model import load_model
####################################################################
# Set environment variable for compatibility issues
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

# Disable unnecessary precision to speed up computations
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
torch.autograd.set_detect_anomaly(False) 
####################################################################


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
        self.n_epochs_stop = 50  # Number of epochs to wait before stopping

    def setup_device(self, args):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"CUDA Status : {self.device.type}")

    def setup_datasets(self, args):
        transform = {
            'train': transforms.Compose([
                transforms.RandomPerspective(distortion_scale=0.5, p=0.5),
                # transforms.RandomRotation(25),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.5),
                transforms.Resize((512, 512)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5], std=[0.5])
            ]),
            'valid': transforms.Compose([
                transforms.Resize((512, 512)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5], std=[0.5])
            ])
        }

        img_dirs = {'train': '/mnt/HDD/octc/seg_data/train_img', 'valid': '/mnt/HDD/octc/seg_data/valid_img' }
        mask_dir = {'train': '/mnt/HDD/octc/seg_data/train_mask', 'valid': '/mnt/HDD/octc/seg_data/valid_mask'} 
        
        tr_dataset = CustomDataset(image_dir = img_dirs['train'], mask_dir = mask_dir['train'], transform=transform['train'], testing=False)
        val_dataset = CustomDataset(image_dir = img_dirs['valid'], mask_dir = mask_dir['valid'], transform=transform['valid'], testing=True)
        self.train_loader = DataLoader(
            tr_dataset,
            batch_size=args.ts_batch_size, shuffle=True
        )
        self.valid_loader = DataLoader(
            val_dataset, 
            batch_size=args.vs_batch_size, shuffle=True
        )
        print("DataLoader Setting Complete.")
        print(f"Train Data : {len(tr_dataset)} 개")
        print(f"Valid Data : {len(val_dataset)} 개")


    def setup_wandb(self, args):
        self.w = args.wandb.strip().lower()
        if args.wandb.strip().lower() == "yes":
            wandb.init(project='oci-inpainting-segmentation', entity='dablro1232', notes='baseline', config=args.__dict__)
            wandb.run.name = args.model + f'_{args.version}_{args.training_date}'
            self.run_name = args.model + f'_{args.version}_{args.training_date}'
        else:
            self.run_name = args.model + '_debug'

    def initialize_models(self, args):
        model_loader = load_model.segmentation_models_loader(model_name = args.model, width = 512, height = 512)
        self.model = model_loader().to(self.device)
                
        self.setup_optimizers(args)
    
    def setup_train(self, args):
        # self.loss_fn = nn.BCEWithLogitsLoss().to(self.device) # ~v7 BCEWithLogitsLoss
        from monai.losses import DiceCELoss
        self.bce_fn = nn.BCEWithLogitsLoss().to(self.device)
        self.dice_fn = DiceCELoss(to_onehot_y = False, sigmoid=True).to(self.device)
        
    def loss_fn(self, outputs, masks):
        bce_loss = self.bce_fn(outputs, masks)
        dice_loss = self.dice_fn(outputs, masks)
        return 0.5 * bce_loss + 0.5 * dice_loss
    def setup_optimizers(self, args):
        self.epochs = args.epochs
        self.optimizer = optim.AdamW(self.model.parameters(), lr=args.learning_rate, betas=(0, 0.9))
        if args.pretrain == 'yes':
            checkpoint = torch.load(args.pretrained_model, map_location=self.device)
            self.best_valid_loss = checkpoint['valid_loss']
            self.load_checkpoints(checkpoint)
            self.epoch = checkpoint['epoch']
            
            print('\033[41m',"#"*30, ' | ', 'Pretrained Setting Complete !!', '\033[0m')
        else:
            self.epoch = 1
            
        self.setup_scheduler(args)  # 스케줄러 설정 호출
    
    def setup_scheduler(self, args):
        # ReduceLROnPlateau 스케줄러 설정
        if args.scheduler == 'ReduceLROnPlateau':
            self.scheduler = ReduceLROnPlateau(self.optimizer, mode='min', factor=0.1, patience=10, verbose=True)
        elif args.scheduler == 'LambdaLR':
            self.scheduler = optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lambda epoch: 0.95 ** epoch)
        else:
            self.scheduler = None
    def load_checkpoints(self, checkpoint):
        self.model_weights = checkpoint['model_state_dict']
        self.model.load_state_dict(self.model_weights)
        self.optim_weights = checkpoint['optimzier_state_dict']
        self.optimizer.load_state_dict(self.optim_weights) #Only use pretrained G optimizer checkpoint 

    def setup_paths(self, args):
        self.save_path = os.path.join(args.save_path, f"{self.run_name}")
        os.makedirs(self.save_path, exist_ok=True)
        save_args(f"{self.save_path}/{self.run_name}.json")
        
        if args.pretrain == "yes":
            print(f"Pretrained model loaded : {args.pretrained_model}")
            self.save_path = args.pretrained_model.split('/')[:-1]
            self.save_path = '/'.join(self.save_path)
            print(self.save_path)
            save_args(f"{self.save_path}/{self.run_name}.json")

    def fit(self):
        for epoch in tqdm(range(self.epoch, self.epochs+1)):
            train_losses, train_accs, train_ious, train_dices, train_hds = 0, 0, 0, 0, 0
            valid_losses, valid_accs, valid_ious, valid_dices, valid_hds = 0, 0, 0, 0, 0
            # Training phase
            self.model.train()
            for images, masks in self.train_loader:
                images, masks = images.to(self.device), masks.to(self.device)
                outputs = self.model(images)

                # BCE Logistic Loss
                
                loss = self.loss_fn(outputs, masks)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                preds = torch.sigmoid(outputs)
                iou, acc, dice, hd = calculate_metrics(preds, masks, threshold=0.5)

                train_losses += loss.cpu().detach().item()
                train_ious += iou
                train_accs += acc
                train_dices += dice
                train_hds += hd
                
            # Validation phase
            with torch.no_grad():
                self.model.eval()
                for (images, masks, paths) in self.valid_loader:
                    images, masks = images.to(self.device), masks.to(self.device)
                    outputs = self.model(images)
                    # BCE Logistic Loss
                    loss = self.loss_fn(outputs, masks).cpu().detach().item()

                    preds = torch.sigmoid(outputs)
                    iou, acc, dice, hd = calculate_metrics(preds, masks, threshold=0.5)
                    valid_losses += loss
                    valid_ious += iou
                    valid_accs += acc
                    valid_dices += dice
                    valid_hds += hd
                    
            if self.scheduler:
                self.scheduler.step(valid_losses)
                
            self.log_metrics(epoch, train_losses, train_accs, train_ious, train_dices, train_hds, valid_losses, valid_accs, valid_ious, valid_dices, valid_hds)
            self.visualize(epoch = epoch, image = images[0,0], mask = masks[0,0], output_image = (preds[0,0] > 0.5).float())
            
            # Early Stopping Check
            if valid_losses < self.best_valid_loss:
                self.best_valid_loss = valid_losses
                self.epochs_no_improve = 0
            else:
                self.epochs_no_improve += 1

            if self.epochs_no_improve >= self.n_epochs_stop:
                self.epochs_no_improve = 0
                self.save_model(epoch, valid_losses/len(self.valid_loader))  # Save the best model
                print(f"\033[41m Early stopping at epoch {epoch}\033[0m")
                print(f"\033[41m Best valid loss: {self.best_valid_loss}\033[0m")
                print(f"\033[41m ACC : {valid_accs / len(self.valid_loader)}\033[0m")
                print(f"\033[41m mIoU: {valid_ious / len(self.valid_loader)}\033[0m")
                print(f"\033[41m mDice: {valid_dices / len(self.valid_loader)}\033[0m")
                print(f"\033[41m mHD: {valid_hds / len(self.valid_loader)}\033[0m")
                # break
            
            if epoch % 50 == 0:
                self.save_model(epoch, valid_losses/len(self.valid_loader))

        if self.w == "yes":
            wandb.finish()
        print("Training Complete.")

        
    def log_metrics(self, epoch, train_losses, train_accs, train_ious, train_dices, train_hds, valid_losses, valid_accs, valid_ious, valid_dices, valid_hds):
        if self.w == 'yes':
            wandb.log({
                "train_loss": train_losses / len(self.train_loader),
                "train_acc": train_accs / len(self.train_loader),
                "train_ious": train_ious / len(self.train_loader),
                "train_dice": train_dices / len(self.train_loader),
                "train_hausdorff_distance": train_hds / len(self.train_loader),
                "valid_loss": valid_losses / len(self.valid_loader),
                "valid_acc": valid_accs / len(self.valid_loader),
                "valid_ious": valid_ious / len(self.valid_loader),
                "valid_dice": valid_dices / len(self.valid_loader),
                "valid_hausdorff_distance": valid_hds / len(self.valid_loader),
                'learning_rate': self.optimizer.param_groups[0]['lr']
            }, step = epoch)

    def save_model(self, epoch, valid_loss):
        current_path = os.path.join(self.save_path, f"model_{epoch}.pt")
        torch.save({
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimzier_state_dict": self.optimizer.state_dict(),
            "valid_loss": valid_loss
        }, current_path)
        print(f"Saved model at {current_path}")

    def visualize(self, epoch, image, mask, output_image):
        current_path = os.path.join(self.save_path, f"result_{epoch}.png")

        vis_imgs = [image, mask, output_image]
        vis_labels = ['Input','Mask(GT)','Result']
        plt.figure(figsize = (12, 8))
        for index, plot_img in enumerate(vis_imgs):
            plt.subplot(1,3,index+1)
            plt.imshow(plot_img.cpu().detach().numpy(), cmap = 'gray')
            plt.title(vis_labels[index])
            plt.axis('off')
        plt.tight_layout()
        plt.savefig(current_path)
        plt.close()
        
        print(f"Visualize save : {epoch}/{self.epochs}")
        