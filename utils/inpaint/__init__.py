import matplotlib.pyplot as plt 
import matplotlib
import cv2
import numpy as np 
import os 
# matplotlib의 Backend를 TkAgg로 설정

def plotting(images, masks, input_images):
    plt.figure(dpi =256)
    plt.subplot(131)
    plt.imshow(images[1,0].cpu().detach().numpy(), cmap= 'gray')
    plt.axis('off')
    plt.title('image')
    plt.subplot(132)
    plt.imshow(masks[1,0].cpu().detach().numpy(), cmap= 'gray')
    plt.axis('off')
    plt.title('mask')
    plt.subplot(133)
    plt.imshow(input_images[1,0].cpu().detach().numpy(), cmap= 'gray')
    plt.axis('off')
    plt.title('Result')
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
    
    
def save_plotting(pred_image, comp_ext, image_path):
    """ 원본 이미지 사이즈로 바꾸기 및 저장하기 """
    original_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    pred_image_np = pred_image.cpu().detach().numpy()
    pred_image_np = np.transpose(pred_image_np, (1, 2, 0))  # (C, H, W) -> (H, W, C)
    
    # 원본 이미지 크기로 resize
    resize_pred_image = cv2.resize(pred_image_np, (original_image.shape[1], original_image.shape[0]))
    resize_pred_image = np.clip(resize_pred_image * 255, 0, 255).astype(np.uint8)  # Scale to 0-255 and convert to uint8
    
    # 결과 저장
    save_path = os.path.join(comp_ext, image_path.split('/')[-1])
    cv2.imwrite(save_path, resize_pred_image)



def visualize_gui(original_images, masks, results):
    matplotlib.use('TkAgg')

    plt.ion()  # Interactive mode on
    plt.figure(figsize=(12, 4))
    titles = ['Original Image', 'Mask', 'Composite Image']

    # 원본 이미지 표시
    plt.subplot(1, 3, 1)
    plt.title(titles[0])
    plt.imshow(original_images[0].cpu().detach().permute(1, 2, 0))
    plt.title('INPUT')
    plt.axis('off')

    # 마스크 표시
    plt.subplot(1, 3, 2)
    plt.title(titles[1])
    plt.imshow(masks[0].cpu().detach().squeeze(), cmap='gray')
    plt.title('MASK')
    plt.axis('off')

    # 복원된 이미지 표시
    plt.subplot(1, 3, 3)
    plt.title(titles[2])
    plt.imshow(results[0].cpu().detach().permute(1, 2, 0))
    plt.axis('off')
    plt.title('OUTPUT')
    plt.show()
    plt.pause(0.1)  # GUI 창이 업데이트되도록 잠시 대기

    plt.ioff()  # Interactive mode off