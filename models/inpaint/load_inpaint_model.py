
import torch
import sys 

from models.inpaint import vae, unet, ocigan 

class inpainting_models_loader:
    def __init__(self, model_name, width, height):
        self.model_name = model_name
        self.width = width
        self.height = height
    def __call__(self):
        return self.load_model()

    def load_model(self):
        if self.model_name == 'VAE':
            model = vae.VAE(self.width, self.height)
            print("Inpaint Model: VAE loaded successfully!!")
        elif self.model_name == 'unet':
            model = unet.UNET(self.width, self.height)
            print("Inpaint Model: UNet loaded successfully!!")
        elif self.model_name == 'OCI-GAN':
            model = ocigan.OCIGAN(self.width, self.height)
            print("Inpaint Model: OCI-GAN loaded successfully!!")
        else:
            raise ValueError('Model name is not valid')
        return model
