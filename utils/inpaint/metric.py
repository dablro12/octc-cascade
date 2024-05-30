import numpy as np
from skimage.metrics import structural_similarity, peak_signal_noise_ratio
from utils.inception import InceptionV3
import torch
from skimage.metrics import structural_similarity, peak_signal_noise_ratio
import csv 

def compare_mae(real, fake):
    real, fake = real.astype(np.float32), fake.astype(np.float32)
    return np.sum(np.abs(real - fake)) / np.sum(real + fake)

def compare_mse(real, fake):
    real, fake = real.astype(np.float32), fake.astype(np.float32)
    return np.mean((real - fake) ** 2)

def compare_psnr(real, fake):
    return peak_signal_noise_ratio(real, fake)

def compare_ssim(real, fake):
    return structural_similarity(real, fake, data_range=1.0, multichannel=False)

# 학습 루프 끝부분에 메트릭을 계산하고 기록하는 코드 추가
def calculate_and_log_metrics(reals, fakes):
    # reals, fakes: 실제 이미지와 생성된 이미지의 numpy 배열 리스트
    mae_value = compare_mae(reals, fakes)
    mse_value = compare_mse(reals, fakes)
    psnr_value = compare_psnr(reals, fakes)
    ssim_value = compare_ssim(reals, fakes)
    return mae_value, mse_value, psnr_value, ssim_value


    
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input
from scipy.linalg import sqrtm
from PIL import Image

def compare_fid(reals, fakes):
    reals, fakes = reals.permute(0,2,3,1).cpu().detach().numpy(), fakes.permute(0,2,3,1).cpu().detach().numpy()
    reals, fakes = (reals * 255).astype(np.uint8), (fakes * 255).astype(np.uint8)
    reals_resized = np.array([np.array(Image.fromarray(image).resize((299, 299))) for image in reals])
    fakes_resized = np.array([np.array(Image.fromarray(image).resize((299, 299))) for image in fakes])
    reals_preprocessed = preprocess_input(reals_resized)
    fakes_preprocessed = preprocess_input(fakes_resized)
    return calculate_fid(reals_preprocessed, fakes_preprocessed)


def calculate_fid(images1, images2):
    # Load InceptionV3 model pre-trained on ImageNet
    model = InceptionV3(include_top=False, pooling='avg', input_shape=(299, 299, 3))

    # Preprocess images
    images1_preprocessed = preprocess_input(images1)
    images2_preprocessed = preprocess_input(images2)

    # Get feature representations
    features1 = model.predict(images1_preprocessed)
    features2 = model.predict(images2_preprocessed)

    # Calculate mean and covariance statistics
    mu1, sigma1 = np.mean(features1, axis=0), np.cov(features1, rowvar=False)
    mu2, sigma2 = np.mean(features2, axis=0), np.cov(features2, rowvar=False)

    # Calculate FID score
    ssdiff = np.sum((mu1 - mu2)**2)
    covmean = sqrtm(sigma1.dot(sigma2))
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    fid = ssdiff + np.trace(sigma1 + sigma2 - 2*covmean)
    return fid

def save_csv(csv_file_path, metric):
    # CSV 파일에 저장
    with open(csv_file_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        
        # CSV 파일의 헤더 쓰기
        writer.writerow(['Metric', 'Value'])
        
        # 각 항목을 행으로 쓰기
        for key, value in metric.items():
            writer.writerow([key, value])