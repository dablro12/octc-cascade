from torch.utils.data import Dataset, BatchSampler
import os 
from PIL import Image
import numpy as np
import torch
import torchvision.transforms.functional as F
import random
import cv2

class CustomDataset(Dataset):
    """CustomDataset with support of transforms for MGGAN.
    Args:
        original image + Binary Mask image
    Returns:
        masked_image, mask, label
    """
    def __init__(self, image_dir, mask_dir, transform = None, mask_transform = None, testing = False, mask_shuffle = False): #transform 가지고올거있으면 가지고 오기 
        self.image_dir = image_dir 
        self.mask_dir = mask_dir
        self.img_transform = transform
        self.mask_transform = mask_transform
        self.mask_paths = []
        self.image_paths = []
        self.testing = testing
        self.mask_shuffle = mask_shuffle
        
        for filename in os.listdir(self.image_dir):
            image_path = os.path.join(self.image_dir, filename)
            self.image_paths.append(image_path)
        for filename in os.listdir(self.mask_dir):
            mask_path = os.path.join(self.mask_dir, filename)
            self.mask_paths.append(mask_path)

    def __len__(self):
        return len(self.image_paths) 
    
    def get_paths(self):
        return self.image_paths
    
    def __getitem__(self, idx):
        # mask_path를 가지고 올때 image_path와 같은 이름으로 가지고 오지 않고 랜덤으로 가지고 오고싶음.
        image_path = self.image_paths[idx]
        if self.mask_shuffle:
            random_mask_idx = random.randint(0, len(self.mask_paths)-1)
            mask_path = self.mask_paths[random_mask_idx]
        else:
            mask_path = self.mask_paths[idx]
            
        #이미지 open to PIL : pytorch는 PIL 선호
        # opencv bgr -> rgb 로 변환
        self.image = Image.open(image_path).convert('RGB') #기존 0~255 3channel
        self.mask = Image.open(mask_path).convert('L') #기존 0~255 1channel
        
        # 입력사이즈에 맞게 resize
        self.image = self.image.resize((512, 512), Image.NEAREST)
        self.mask = self.mask.resize((512, 512), Image.NEAREST)
        
        self.image = self.img_transform(self.image)
        self.mask = self.mask_transform(self.mask)
        # 테스트시 이미지 경로 가지고옴
        if self.testing:
            return self.image, self.mask, image_path
        else: 
            return self.image, self.mask 

        # print(self.image, self.mask)
        # 위아래 제외하고 crop 해놓기 -> 원본과 같은사이즈로
class Test_CustomDataset(Dataset):
    """CustomDataset with support of transforms for MGGAN.
    Args:
        original image + Binary Mask image
    Returns:
        masked_image, mask, label
    """
    def __init__(self, image_dir, mask_dir, transform = None, mask_transform = None,): #transform 가지고올거있으면 가지고 오기 
        self.image_dir = image_dir 
        self.mask_dir = mask_dir
        self.img_transform = transform
        self.mask_transform = mask_transform
        self.mask_paths = []
        self.image_paths = []
        
        for filename in os.listdir(self.image_dir):
            image_path = os.path.join(self.image_dir, filename)
            self.image_paths.append(image_path)
        for filename in os.listdir(self.mask_dir):
            mask_path = os.path.join(self.mask_dir, filename)
            self.mask_paths.append(mask_path)



    def __len__(self):
        return len(self.image_paths) 
    
    def get_paths(self):
        return self.image_paths
    
    def __getitem__(self, idx):
        # mask_path를 가지고 올때 image_path와 같은 이름으로 가지고 오지 않고 랜덤으로 가지고 오고싶음.
        image_path = self.image_paths[idx]
        mask_path = self.mask_paths[idx]
            
        #이미지 open to PIL : pytorch는 PIL 선호
        # opencv bgr -> rgb 로 변환
        self.image = Image.open(image_path).convert('RGB') #기존 0~255 3channel
        self.mask = Image.open(mask_path).convert('L') #기존 0~255 1channel
        
        # 입력사이즈에 맞게 resize
        self.image = self.image.resize((512, 512), Image.NEAREST)
        self.mask = self.mask.resize((512, 512), Image.NEAREST)
        
        self.image = self.img_transform(self.image)
        if self.mask_transform != None:
            self.mask = self.img_transform(self.mask)
        
        return self.image, self.mask, image_path
    
    
class BalancedBatchSampler(BatchSampler):
    def __init__(self, dataset, batch_size):
        self.dataset = dataset
        self.batch_size = batch_size
        
        # class 에 따라 구분
        self.class0_indices = [i for i, (_, label) in enumerate(dataset) if label == 0]
        self.class1_indices = [i for i, (_, label) in enumerate(dataset) if label == 1]

    def __iter__(self):
        random.shuffle(self.class0_indices)
        random.shuffle(self.class1_indices)

        # 반반씩 뽑기 위한 배치 사이즈 조정
        half_batch = self.batch_size // 2

        for i in range(0, min(len(self.class0_indices), len(self.class1_indices)), half_batch):
            batch_indices = []
            batch_indices.extend(self.class0_indices[i:i + half_batch])
            batch_indices.extend(self.class1_indices[i:i + half_batch])
            random.shuffle(batch_indices)  # 배치 내부 셔플
            yield batch_indices

    def __len__(self):
        return min(len(self.class0_indices), len(self.class1_indices)) // (self.batch_size // 2)
