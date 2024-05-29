
import torch
import os 

class models_metric_loader:
    def __init__(self, vae_path, ocigan_path, unet_path):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.ocigan = self.ocigan_model_load(
            load_path = ocigan_path
        )
        self.vae = self.vae_model_load(
            load_path = vae_path
        )
        self.unet = self.unet_model_load(
            load_path = unet_path
        )
        
    def set(self):
        return self.ocigan, self.vae, self.unet

    def vae_model_load(self, load_path):
        from model import VAE
        input_dim = 4
        hidden_dim =512
        latent_dim = 32
        n_embeddings = 512
        output_dim = 3

        encoder = VAE.Encoder(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=latent_dim)
        codebook = VAE.VQEmbeddingEMA(n_embeddings=n_embeddings, embedding_dim=latent_dim)
        decoder = VAE.Decoder(input_dim=latent_dim, hidden_dim=hidden_dim, output_dim=output_dim)

        model = VAE.Model(Encoder=encoder, Codebook=codebook, Decoder=decoder).to(self.device)

        G_check = torch.load(load_path)['netG_state_dict']
        model.load_state_dict(G_check)
        print("#"*30, 'VAE Setup Complete', "#"*30)
        return model 

    def ocigan_model_load(self, load_path):
        from model import aotgan 
        from model.aotgan import InpaintGenerator
        # oci-gan
        # device = 'cpu'
        netG = InpaintGenerator().to(self.device)
        G_check = torch.load(load_path)['netG_state_dict']
        netG.load_state_dict(G_check)
        print("#"*30, 'OCI-GAN Setup Complete', "#"*30)
        return netG


    def unet_model_load(self, load_path):
        from model import unet
        input_dim = 4
        output_dim = 3
        model = unet.GeneratorUNet(in_channels=input_dim, out_channels= output_dim).to(self.device)
        G_check = torch.load(load_path)['netG_state_dict']
        model.load_state_dict(G_check)
        print("#"*30, 'Unet Setup Complete', "#"*30)
        return model 
