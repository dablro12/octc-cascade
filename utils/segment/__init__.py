import matplotlib.pyplot as plt 
import torch 
import os 
import numpy as np 
import cv2

# matplotlib의 Backend를 TkAgg로 설정
def train_plotting(images, masks):
    plt.figure(dpi =256)
    plt.subplot(121)
    plt.imshow(images[1,0], cmap= 'gray')
    plt.title('image')
    plt.subplot(122)
    plt.imshow(masks[1,0], cmap= 'gray')
    plt.title('mask')
    plt.tight_layout()
    plt.show()
    
def test_plotting(input_image, mask, pred_image, save_path):
    plt.figure(dpi =256)
    plt.subplot(131)
    plt.imshow(input_image.permute(1, 2, 0).cpu().detach().numpy(), cmap= 'gray')
    plt.axis('off')
    plt.title('Input')
    plt.subplot(132)
    plt.imshow(mask.permute(1, 2, 0).cpu().detach().numpy(), cmap= 'gray')
    plt.axis('off')
    plt.title('Mark')
    plt.subplot(133)
    plt.imshow(pred_image.permute(1, 2, 0).cpu().detach().numpy(), cmap= 'gray')
    plt.title('Output')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def save_validation(input_images, masks, output_images, epoch, save_dir, threshold=0.5,):
    plt.figure(dpi=128)
    plt.subplot(131)
    plt.imshow(input_images[0, 0].cpu().detach().numpy(), cmap='gray')
    plt.title("Input")
    plt.axis('off')
    plt.subplot(132)
    plt.imshow(masks[0, 0].cpu().detach().numpy(), cmap='gray')
    plt.title("mask")
    plt.axis('off')
    plt.subplot(133)
    plt.imshow((output_images[0, 0]>threshold).int().cpu().detach().numpy(), cmap='gray')
    plt.title('Output')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'epoch_{epoch}.png'))
    plt.show()
    plt.close()
    
def save_comparision(input_images, masks, output_images, filenames, save_dir, threshold=0.5):
    """ Save the comparison of input, mask, and output images."""
    for i in range(len(input_images)):
        vis_imgs = [input_images[i,0], masks[i,0], (output_images[i,0] > threshold).int()]
        vis_labels = ['Input','Mask(GT)','Result']
        plt.figure(figsize = (12, 8))
        for index, plot_img in enumerate(vis_imgs):
            plt.subplot(1,3,index+1)
            plt.imshow(plot_img.cpu().detach().numpy(), cmap = 'gray')
            plt.title(vis_labels[index])
            plt.axis('off')
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f'{filenames[i]}'))
        plt.close()
    

def save_prediction(preds, threshold, paths, save_dir):
    """ Save the output images(png). & setting Hyperparameters Threshold """
    
    filenames = [os.path.basename(path) for path in paths]  # paths에서 파일명 추출
    
    # 마스크 사이즈를 원본 사이즈에 맞게 저장 
    for i in range(len(preds)):
        # 원본 이미지 상지ㅡ로 resize
        original_size = cv2.imread(paths[i], cv2.IMREAD_GRAYSCALE).shape
        resized_mask = cv2.resize((preds[i].cpu().detach() > threshold).int().squeeze().numpy(), (original_size[1], original_size[0]), interpolation=cv2.INTER_NEAREST)
        plt.imsave(os.path.join(save_dir, filenames[i]), resized_mask, cmap='gray')
    

def save_model(model, optimizer, epoch, save_dir):
    torch.save({
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
    }, os.path.join(save_dir, f'epoch_{epoch}.pt'))
    
def save_metrics(metrics, save_path):
    import pandas as pd 
    """ dictonary 형태 metrics을 dataframe으로 저장 """
    df = pd.DataFrame(metrics)
    df.to_csv(save_path, index=False)
    
def save_loss(metrics, save_dir):
    # loss plot
    plt.figure(dpi=128)
    for key, value in metrics.items():
        plt.plot(value, label=key)
    plt.legend()
    plt.savefig(os.path.join(save_dir, 'loss.png'))
    plt.close()
    np.save(os.path.join(save_dir, 'metrics.npy'),metrics)
