from utils.metric import calculate_and_log_metrics, compare_fid
import torch 
import pandas as pd 
import numpy as np 
import cv2 
import os 
from utils import save_plotting
class metric_models:
    def __init__(self, test_loader, ocigan, vae, unet):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.test_loader = test_loader
        self.total_samples = len(self.test_loader.dataset)
        self.ocigan, self.vae, self.unet = ocigan, vae, unet  
        
    def metric_forward(self):
        oci_metric = self.test_metric(self.ocigan, model_name = 'ocigan')
        vae_metric = self.test_metric(self.vae, model_name = 'vae')
        unet_metric = self.test_metric(self.unet, model_name = 'unet')
        return oci_metric, vae_metric, unet_metric        
    def inference_forward(self):
        self.test_inference(self.ocigan, model_name = 'ocigan')
        self.test_inference(self.vae, model_name = 'vae')
        self.test_inference(self.unet, model_name = 'unet')
            
    def test_metric(self, model, model_name):
        metric, mae_sum, mse_sum, psnr_sum ,ssim_sum = self.init_metric()
        with torch.no_grad():
            model.eval()
            for images, masks, image_paths in self.test_loader:
                images, input_images, masks = self.prepare_images(images, masks)
                if model_name == 'ocigan':
                    pred_images = model(input_images, masks) #Need to 3+1 channel 
                if model_name == 'vae':
                    pred_images, commitment_loss, codebook_loss, perplexity = model(input_images, masks) #Need to 3+1 channel 
                if model_name == 'unet':
                    pred_images= model(input_images, masks)
                # if model_name == 'cyclegan':
                    # TODO

                comp_images = self.compute_composite_images(input_images, pred_images, masks)
                
                # metric를 평가하기 위해 metric_images와 pred_images에서 mask와 같은 인덱스인 부분을 제외하고 0으로 만들기 
                images, comp_images = images.clone(), pred_images.clone()
                images[masks.repeat(1,3,1,1) == 0] = 0
                comp_images[masks.repeat(1,3,1,1) == 0] = 0
                
                # metric 계산 : mask인 부분만 계산! 
                for i in range(images.size(0)):
                    mae, mse, psnr, ssim = calculate_and_log_metrics(images[i,0].cpu().detach().numpy(), comp_images[i,0].cpu().detach().numpy())
                    mae_sum += mae 
                    mse_sum += mse 
                    psnr_sum += psnr
                    ssim_sum += ssim
                fid = compare_fid(images, comp_images) 
                
                metric['mae'] += mae_sum / self.total_samples
                metric['mse'] += mse_sum / self.total_samples
                metric['psnr'] += psnr_sum / self.total_samples
                metric['ssim'] += ssim_sum / self.total_samples
                metric['fid'] += fid / len(self.test_loader)
        return metric

    def test_inference(self, model, model_name):
        save_dir = self.init_test_inference(model_name)
        with torch.no_grad():
            model.eval()
            for images, masks, image_paths in self.test_loader:
                images, input_images, masks = self.prepare_images(images, masks)
                if model_name == 'ocigan':
                    pred_images = model(input_images, masks) #Need to 3+1 channel 
                if model_name == 'vae':
                    pred_images, commitment_loss, codebook_loss, perplexity = model(input_images, masks) #Need to 3+1 channel 
                if model_name == 'unet':
                    pred_images= model(input_images, masks)
                # if model_name == 'cyclegan':
                    # TODO
                comp_images = self.compute_composite_images(input_images, pred_images, masks)

                for comp_image, image_path in zip(comp_images, image_paths):
                    # 원본 이미지 사이즈로 바꾸기
                    original_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
                    comp_image_np = comp_image.cpu().detach().numpy()
                    comp_image_np = np.transpose(comp_image_np, (1, 2, 0))  # (C, H, W) -> (H, W, C)
                    
                    # 원본 이미지 크기로 resize
                    resize_comp_image = cv2.resize(comp_image_np, (original_image.shape[1], original_image.shape[0]))
                    resize_comp_image = np.clip(resize_comp_image * 255, 0, 255).astype(np.uint8)  # Scale to 0-255 and convert to uint8
                    
                    # 결과 저장
                    save_path = os.path.join(save_dir, image_path.split('/')[-1])
                    cv2.imwrite(save_path, resize_comp_image)
        print(f"Saved resized prediction image at {save_dir}")
        

    def init_test_inference(self, model_name):
        save_dir = os.path.join('/mnt/HDD/oci_models/models/classification_data', model_name)
        os.makedirs(save_dir, exist_ok= True) #<-- 없으면 생성
        return save_dir 
    
    def prepare_images(self, images, masks):
        input_images = images.clone()
        input_images[masks != 0] = masks[masks != 0]
        return images.to(self.device), input_images.to(self.device), masks[:,0,:,:].unsqueeze(1).to(self.device)

    def compute_composite_images(self, input_images, pred_images, masks):
        comp_images = input_images.clone()
        comp_images[masks.repeat(1,3,1,1) != 0] = pred_images[masks.repeat(1,3,1,1) != 0]
        return comp_images
    
    
    def init_metric(self):
        metric = {'mae' : 0, 'mse' : 0,  'psnr' : 0, 'ssim' : 0, 'fid' : 0}
        mae_sum, mse_sum, psnr_sum, ssim_sum = 0, 0, 0, 0
        return metric, mae_sum, mse_sum, psnr_sum, ssim_sum 
    