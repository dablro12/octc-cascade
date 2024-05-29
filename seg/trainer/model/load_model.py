import sys 
import torch.nn as nn

from model import models
from model import unet, unet_plus_plus, manet, swinunet

class segmentation_models_loader:
    def __init__(self, model_name, width, height):
        self.model_name = model_name
        self.width = width
        self.height = height
    def __call__(self):
        return self.load_model()

    def load_model(self):
        if self.model_name == 'unet':
            model = unet.U_Net(self.width, self.height)
            print("Model: U-Net loaded successfully!! | pretrained : imagenet")
        elif self.model_name == 'r2unet':
            model = models.R2U_Net(self.width, self.height)
            print("Model: R2U_Net loaded successfully!! | pretrained : False")
        elif self.model_name == 'attunet':
            model = models.AttU_Net(self.width, self.height)
            print("Model: AttU_Net loaded successfully!! | pretrained : False")
        elif self.model_name == 'r2attunet':
            model = models.R2AttU_Net(self.width, self.height)
            print("Model: R2AttU_Net loaded successfully!! | pretrained : False")
        elif self.model_name == 'unet_plus_plus':
            model = unet_plus_plus.unet_plus_plus(self.width, self.height)
            print("Model: Unet++ loaded successfully!! | pretrained : imagenet")
        elif self.model_name == 'manet':
            model = manet.manet(self.width, self.height)
            print("Model: MAnet loaded successfully!! | pretrained : imagenet")
        elif self.model_name == 'swinunet':
            model = swinunet.swinunet(self.width, self.height)
            print("Model: swinunet loaded successfully!! | pretrained : None")
        elif self.model_name == 'monai_swinunet':
            model = swinunet.monai_swinunet(self.width, self.height)
            print("Model: MONAI-SwinUNET loaded successfully!! | pretrained : brain MRI")
        else:
            raise ValueError('Model name is not valid')
        return model