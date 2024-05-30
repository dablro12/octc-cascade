import matplotlib.pyplot as plt
def plot_cascade_result(image, segment_output, inpaint_mask, inpaint_output):
    plt.figure(dpi = 256, figsize= (4,4))
    plt.subplot(221)
    plt.imshow(image, cmap= 'gray')
    plt.title('Marker Image[Input]]')
    plt.subplot(222)
    plt.imshow(segment_output, cmap= 'gray')
    plt.title('Segment Result')
    plt.subplot(223)
    plt.imshow(inpaint_mask, cmap= 'gray')
    plt.title('Inpainting mask')
    plt.subplot(224)
    plt.imshow(inpaint_output, cmap= 'gray')
    plt.title('Inpainting Results')
    plt.tight_layout()
    plt.show()
    
def compute_composite_images(input_images, pred_images, inpaint_masks):
    ## mask에서 0이 아닌 부분을 GT로 대체, 이때 마스크는 0~1사이의 값을 가짐 
    input_images = (input_images + 1) / 2
    comp_images = input_images.clone()
    comp_images[inpaint_masks.repeat(1,3,1,1) != 0] = pred_images[inpaint_masks.repeat(1,3,1,1) != 0]
    
    return comp_images

def mask_preprocessing(input_images, seg_masks):
    # mask 가 -1 ~ 1로 정규화 되어있으므로 사이이므로 0~255로 변환
    # masks = (masks + 1) * 127.5
    # masks[masks != 0] = input_images[masks != 0]
    inpaint_masks = input_images.clone()
    # -1~1로 범위로 정규화 되어있는 inpaint_mask를 0~1 범위로 다시 정규화
    inpaint_masks = (inpaint_masks + 1) / 2

    # 마스크가 0인 부분을 제외하고 1로 변경
    seg_masks[seg_masks != 0] = 1
    # 이미지와 마스크를 곱해서 배경을 제거
    inpaint_masks = inpaint_masks * seg_masks

    # inpaint_masks[seg_masks == 0] = 0
    # inpaint_masks에서 seg_mask가 1인 부분 제외하고 0으로 채우기
    
    input_images = input_images.repeat(1,3,1,1) #입력값에 맞춰주기 위함 
    return input_images, inpaint_masks

def prepare_images(self, images, masks):
    input_images = images.clone()
    input_images[masks != 0] = masks[masks != 0]
    return images.to(self.device), input_images.to(self.device), masks[:,0,:,:].unsqueeze(1).to(self.device)
