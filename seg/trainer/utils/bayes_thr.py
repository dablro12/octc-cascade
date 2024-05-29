from skopt import gp_minimize
from skopt.space import Real
import numpy as np 
import torch

def objective(threshold, model, data_loader, device):
    """ 
    Args:
        threshold (float): 이진화 임계값
        model (torch.nn.Module): 모델
        data_loader (torch.utils.data.DataLoader): 데이터 로더
        device (torch.device): 디바이스
    Returns:
        float: 최적의 threshold에 대한 평균 IoU
    """
    ious = []
    for images, masks in data_loader:
        images, masks = images.to(device), masks.to(device)
        with torch.no_grad():
            outputs = model(images)
        preds_binary = (outputs > threshold).int()
        iou = ((preds_binary & masks).sum((1, 2, 3)) / (preds_binary | masks).sum((1, 2, 3))).mean().item()
        ious.append(iou)
    avg_iou = np.mean(ious)
    return -avg_iou  # Minimize negative IoU to maximize IoU

def find_best_threshold_bayesian(model, data_loader, device):
    """
    베이지 정리를 사용하여 최적의 threshold를 찾는 함수
    Args:
        model (torch.nn.Module): 모델
        data_loader (torch.utils.data.DataLoader): 데이터 로더
        device (torch.device): 디바이스
    Returns:
        float: 최적의 threshold
    Usage:
        best_threshold = find_best_threshold_bayesian(model, valid_loader, device)
    """
    result = gp_minimize(lambda x: objective(x[0], model, data_loader, device), [Real(0.0, 1.0)], n_calls=30)
    best_threshold = result.x[0]
    best_score = -result.fun
    print(f'Best threshold: {best_threshold}, Best IoU: {best_score}')
    return best_threshold

