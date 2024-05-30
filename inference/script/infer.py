
import sys 
import os 
project_root = os.path.abspath('/home/eiden/eiden/octc-cascade')
sys.path.append(project_root)
#******************************************************************#
import torch
from torch.utils.data import DataLoader 
from torchvision import transforms

import cv2 
import numpy as np 

from models.model_loader import cascade_models_load
from utils.dataset import Inference_Cascade_CustomDataset


class Seg_Inpaint_Service:
    def __init__(self, img_dir, save_dir, segment_model_path, inpaint_model_path, width = 512, height = 512, batch_size=1, seed = 627):
        self.img_dir = img_dir
        self.save_dir = save_dir
        self.segment_model_path = segment_model_path
        self.inpaint_model_path = inpaint_model_path
        # 경로가 디렉토리인지 확인하고, 필요하다면 생성합니다.
        # os.makedirs(self.save_dir, exist_ok=True)
        '''
        # #mysql의 정보를 받아오는 코드입니다.
        self.sql_config_path = sql_config_path
        # with open(self.sql_config_path, 'r') as f:
        #     mysql_config = json.load(f)

        self.host = mysql_config['host']
        self.user = mysql_config['user']
        self.password = mysql_config['password']
        self.database = mysql_config['database']
        '''
        self.width =width
        self.height = height
        self.seed = seed 
        self.batch_size = batch_size
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.load_model() # weight model 설정 및 로드
        self.setup_dataloader()        # 데이터 로더 설정

    def setup_dataloader(self):
        test_transform = transforms.Compose([
            transforms.Resize((self.width, self.height)),
            transforms.ToTensor(),
        ])

        test_dataset = Inference_Cascade_CustomDataset(
            image_dir = self.img_dir,
            transform= test_transform,
            seed = self.seed
        )
        self.test_loader = DataLoader(dataset = test_dataset, batch_size = self.batch_size, shuffle = False)

    def load_model(self):
        cascade_models_loader = cascade_models_load(
            seg_model_path= self.segment_model_path,
            inpaint_model_path= self.inpaint_model_path,
            device = self.device,
            width = self.width,
            height = self.height
        )
        self.seg_model, self.inpaint_model = cascade_models_loader.load_models()
        
    def save_result(self, image_paths, inpaint_outputs):
        '''
        try:
            # MySQL 데이터베이스 연결
            conn = pymysql.connect(
                host=self.host,
                user=self.user,
                password=self.password,
                database=self.database
            )
            cursor = conn.cursor()

            for input_image, mask, pred_image, path in zip(input_images, masks, comp_images, image_paths):
                file_name = self.segment_model_name + '_' + path.split('/')[-1]
                # Origianl Image 대로 Resize
                original_image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
                utils.test_plotting(input_image, mask, pred_image, save_path=os.path.join(self.visual_save_dir, file_name))

                pred_image = pred_image.cpu().detach().numpy()
                # Transpose pred_image from (C, H, W) to (H, W, C) for OpenCV compatibility
                pred_image = np.transpose(pred_image, (1, 2, 0))

                # Now, resize pred_image to match the original image's dimensions
                pred_image = cv2.resize(pred_image, (original_image.shape[1], original_image.shape[0]))  # Note the order of shape
                pred_image = np.clip(pred_image * 255, 0, 255).astype(np.uint8)  # Scale to 0-255 and convert to uint8
                cv2.imwrite(os.path.join(self.save_dir, file_name), pred_image)

                # 결과를 MySQL 데이터베이스에 저장
                query = "UPDATE IMAGEBOARD SET image2 = %s WHERE image = %s"
                values = ('/inpainted/'+ file_name, '/image/'+ file_name)
                cursor.execute(query, values)
                conn.commit()

            print("쿼리 실행 완료")
            print(os.path.join(self.save_dir, file_name))


        except pymysql.Error as e:
            print("MySQL 에러 발생:", e)

        finally:
            # 연결 닫기
            if conn:
                conn.close() 
        '''
        # self.image_paths와 self.data_path를 결합
        data_paths = [os.path.join(self.img_dir, path) for path in image_paths]
        for data_path, inpaint_output in zip(data_paths, inpaint_outputs):
            file_name = data_path.split('/')[-1]
            # Origianl Image 대로 Resize
            original_image = cv2.imread(data_path, cv2.IMREAD_GRAYSCALE)
            
            inpaint_output = inpaint_output.cpu().detach().numpy()
            # Transpose pred_image from (C, H, W) to (H, W, C) for OpenCV compatibility
            inpaint_output = np.transpose(inpaint_output, (1, 2, 0))
            # Now, resize pred_image to match the original image's dimensions
            inpaint_output = cv2.resize(inpaint_output, (original_image.shape[1], original_image.shape[0]))  # Note the order of shape
            inpaint_output = np.clip(inpaint_output * 255, 0, 255).astype(np.uint8)  # Scale to 0-255 and convert to uint8
            cv2.imwrite(os.path.join(self.save_dir, file_name), inpaint_output)
            
        
    def infer(self):
        with torch.no_grad():
            self.seg_model.eval(), self.inpaint_model.eval()
            for images, image_paths in self.test_loader:
                images = images.to(self.device)
                seg_outputs = self.segment_inference(images)# Segmentation Inference
                inpaint_outputs = self.inpaint_inference(images, seg_outputs) # Inpainting Inference
                
                self.save_result(image_paths, inpaint_outputs)# Save Result

    def segment_inference(self, images, threshold = 0.5):
        seg_outputs = self.seg_model(images)
        seg_outputs = torch.sigmoid(seg_outputs)
        seg_outputs = (seg_outputs > threshold).float()
        return seg_outputs
    
    def inpaint_inference(self, images, seg_outputs):
        ### inference
        inpaint_inputs, inpaint_masks = self.mask_preprocessing(images, seg_outputs) 
        pred_images = self.inpaint_model(inpaint_inputs, inpaint_masks) # ocigan
        comp_images = self.compute_composite_images(inpaint_inputs, pred_images, inpaint_masks)
        return comp_images
    
    def compute_composite_images(self, input_images, pred_images, inpaint_masks):
    ## mask에서 0이 아닌 부분을 GT로 대체, 이때 마스크는 0~1사이의 값을 가짐 
        comp_images = input_images.clone()
        comp_images[inpaint_masks.repeat(1,3,1,1) != 0] = pred_images[inpaint_masks.repeat(1,3,1,1) != 0]
        return comp_images

    def mask_preprocessing(self, images, seg_masks):
        # seg_masks를 numpy로 변환후 opencv의 dilation 을 통해 확장한 후 다시 tensor로 변환 
        seg_masks = seg_masks.cpu().detach().numpy().squeeze(1)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))
        dilated_seg_masks = []
        for seg_mask in seg_masks:
            dilated_seg_mask = cv2.dilate(seg_mask, kernel, iterations=4)
            dilated_seg_mask = cv2.erode(dilated_seg_mask, kernel, iterations=2)
            dilated_seg_masks.append(dilated_seg_mask)
        
        # dilated_seg_masks를 다시 numpy 배열로 변환
        dilated_seg_masks = np.array(dilated_seg_masks)
        # 다시 torch 텐서로 변환하고, 원래 디바이스로 이동
        dilated_seg_masks = torch.tensor(dilated_seg_masks).unsqueeze(1).to(images.device)
        
        input_images = images.clone()
        inpaint_masks = images.clone()
        # -1~1로 범위로 정규화 되어있는 inpaint_mask를 0~1 범위로 다시 정규화
        # inpaint_masks = (inpaint_masks + 1) / 2
        # inpaint_masks = inpaint_masks / inpaint_masks.max()
        
        # 이미지와 마스크를 곱해서 배경을 제거
        inpaint_masks = inpaint_masks * dilated_seg_masks

        # input_images = (input_images + 1) / 2
        # input_images = input_images / input_images.max()
        input_images = input_images.repeat(1,3,1,1) # Inpaint Model 입력값에 맞춰주기 위함 
        

        return input_images, inpaint_masks

    
    def run(self):
        self.infer()
        print("Inference Done!")
