import torch
from scipy.spatial.distance import directed_hausdorff
from monai.metrics import DiceMetric
import numpy as np 
def train_metrics(preds, targets, threshold=0.5):
    """
    mIoU 및 Accuracy 계산 함수.
    
    Args:
        preds (torch.Tensor): 모델의 예측값.
        targets (torch.Tensor): 실제 레이블.
        threshold (float): 이진화 임계값.
        
    Returns:
        (tuple): mIoU와 Accuracy의 튜플.
    """
    targets_binary = targets.bool()
    preds_binary = (preds > threshold).bool()
    intersection = (preds_binary & targets_binary).sum((1, 2, 3))
    union = (preds_binary | targets_binary).sum((1, 2, 3))

    # Calculate metrics
    iou = calculate_iou(preds_binary.cpu().numpy(), targets_binary.cpu().numpy())
    accuracy = calculate_mean_accuracy(preds_binary.cpu().numpy(), targets_binary.cpu().numpy())

    return iou, accuracy

def calculate_metrics(preds, targets, threshold=0.5):
    """
    mIoU, Accuracy, Dice Score 및 Hausdorff Distance Metric을 계산하는 함수.
    
    Args:
        preds (torch.Tensor): 모델의 예측값.
        targets (torch.Tensor): 실제 레이블.
        threshold (float): 이진화 임계값.
        
    Returns:
        (tuple): mIoU, Accuracy, Dice Score, Hausdorff Distance의 튜플.
    """
    targets_binary = targets.bool()
    preds_binary = (preds > threshold).bool()
    intersection = (preds_binary & targets_binary).sum((1, 2, 3))
    union = (preds_binary | targets_binary).sum((1, 2, 3))

    # Calculate metrics
    iou = calculate_iou(preds_binary.cpu().numpy(), targets_binary.cpu().numpy())
    accuracy = calculate_mean_accuracy(preds_binary.cpu().numpy(), targets_binary.cpu().numpy())

    # Dice Score
    dice_metric = DiceMetric(include_background=False, reduction='mean')
    dice_score = dice_metric(preds_binary, targets_binary).mean().item()

    # Hausdorff Distance
    hausdorff_distance = calculate_symmetric_hausdorff(preds_binary.cpu().numpy(), targets_binary.cpu().numpy())
    
    return iou, accuracy, dice_score, hausdorff_distance

def calculate_symmetric_hausdorff(preds, targets):
    """ 대칭적인 Hausdorff 거리를 얻기 위해 두 방향 모두를 계산하고 최대값을 사용 """
    hausdorff_distances = []
    for i in range(preds.shape[0]):  # Loop over the batch
        u = preds[i].reshape(-1, preds.shape[1] * preds.shape[2])  # Reshape from H, W to H*W
        v = targets[i].reshape(-1, targets.shape[1] * targets.shape[2])
        forward = directed_hausdorff(u, v)[0]
        backward = directed_hausdorff(v, u)[0]
        hausdorff_distances.append(max(forward, backward))
    return max(hausdorff_distances)  # Return the maximum across the batch

def calculate_iou(pred_mask, true_mask):
    """
    Calculate the Intersection over Union (IoU) for binary masks.
    
    Parameters:
    pred_mask (numpy array): Predicted binary mask
    true_mask (numpy array): Ground truth binary mask
    
    Returns:
    float: IoU score
    """
    intersection = np.logical_and(pred_mask, true_mask).sum()
    union = np.logical_or(pred_mask, true_mask).sum()
    
    if union == 0:
        return 1.0  # Avoid division by zero, consider full overlap if both masks are empty
    else:
        iou = intersection / union
        return iou
    
def calculate_accuracy(pred_mask, true_mask):
    """
    Calculate the mean accuracy for binary masks.
    Parameters:
    pred_mask (numpy array): Predicted binary mask
    true_mask (numpy array): Ground truth binary mask
    
    Returns:
    float: Accuracy score
    """
    # Ensure masks are binary (0 or 1)
    pred_mask = np.round(pred_mask).astype(int)
    true_mask = np.round(true_mask).astype(int)
    
    # Calculate accuracy

    correct_predictions = (pred_mask == true_mask).sum()
    total_pixels = pred_mask.size
    
    accuracy = correct_predictions / total_pixels
    return accuracy


def calculate_mean_accuracy(pred_mask, true_mask):
    """
    Calculate the mean accuracy for binary masks.
    * 백그라운드가 많을 때 사용 
    Parameters:
    pred_mask (numpy array): Predicted binary mask
    true_mask (numpy array): Ground truth binary mask
    
    Returns:
    float: Mean accuracy score
    """
    # Ensure masks are binary (0 or 1)
    pred_mask = np.round(pred_mask).astype(int)
    true_mask = np.round(true_mask).astype(int)
    
    # Calculate accuracy for background (0) and object (1)
    background_mask = (true_mask == 0)
    object_mask = (true_mask == 1)
    
    # Background accuracy
    background_correct = (pred_mask[background_mask] == true_mask[background_mask]).sum()
    background_total = background_mask.sum()
    background_accuracy = background_correct / background_total if background_total > 0 else 1.0
    
    # Object accuracy
    object_correct = (pred_mask[object_mask] == true_mask[object_mask]).sum()
    object_total = object_mask.sum()
    object_accuracy = object_correct / object_total if object_total > 0 else 1.0
    
    # Mean accuracy
    mean_accuracy = (background_accuracy + object_accuracy) / 2
    return mean_accuracy
