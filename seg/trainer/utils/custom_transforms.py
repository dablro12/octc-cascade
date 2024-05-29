
import albumentations as A
from albumentations.pytorch import ToTensorV2
from PIL import ImageFilter
# 사용자 정의 가우시안 블러 적용 함수
class ApplyGaussianBlur(object):
    def __init__(self, radius=2):
        self.radius = radius

    def __call__(self, img):
        return img.filter(ImageFilter.GaussianBlur(self.radius))