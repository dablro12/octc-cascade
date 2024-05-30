import torch
import os 
from matplotlib import pyplot as plt
import numpy as np

class save_class:
    def __init__(self, save_dir):
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)

    def save_validation(self, labels, masks, input_images, pred_images, pred_masks, comp_images,  epoch):
        plt.figure(dpi=128)
        plt.subplot(231)
        plt.imshow(labels[0, 0].cpu().detach().numpy(), cmap='gray')
        plt.title("original")
        plt.subplot(232)
        plt.imshow(masks[0, 0].cpu().detach().numpy(), cmap='gray')
        plt.title("mask")
        plt.subplot(233)
        plt.imshow(input_images[0, 0].cpu().detach().numpy(), cmap='gray')
        plt.title('input image')
        plt.subplot(234)
        plt.imshow(pred_images[0, 0].cpu().detach().numpy(), cmap='gray')
        plt.title('pred before image')
        plt.subplot(235)
        plt.imshow(pred_masks[0, 0].cpu().detach().numpy(), cmap='gray')
        plt.title('pred Discriminator mask')
        plt.subplot(236)
        plt.imshow(comp_images[0, 0].cpu().detach().numpy(), cmap='gray')
        plt.title('Result image')
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, f'epoch_{epoch}.png'))
        plt.show()
        plt.close()

    def save_model(self, netG, netD, optimG, optimD, epoch):
        torch.save({
            'netG': netG.state_dict(),
            'netD': netD.state_dict(),
            'optimG': optimG.state_dict(),
            'optimD': optimD.state_dict()
        }, os.path.join(self.save_dir, f'epoch_{epoch}.pt'))

    def save_loss(self, metrics):
        # loss plot
        plt.figure(dpi=128)
        for key, value in metrics.items():
            plt.plot(value, label=key)
        plt.legend()
        plt.savefig(os.path.join(self.save_dir, 'loss.png'))
        plt.close()
        
        np.save(os.path.join(self.save_dir, 'metrics.npy'),metrics)
