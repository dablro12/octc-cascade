import os 
import torch
from models.inpaint import load_inpaint_model
from models.segment import load_segment_model
class cascade_models_load:
    """
    <사용예시>
    cascade_model_loader = cascade_models_load(
        seg_model_path = '/mnt/HDD/oci-seg_models/monai_swinunet_v3_240530/model_400.pt',
        inpaint_model_path = '/mnt/HDD/oci_models/aotgan/OCI-GAN_v3_240508/model_64.pt',
        # inpaint_model_path = '/mnt/HDD/oci_models/models/VAE_v1_240510/model_27.pt',
        
        device = device
    )
    seg_model, inpaint_model = cascade_model_loader.load_models()
    cascade_model_name = cascade_model_loader.get_cascade_model_name()
    """
    
    """
    <설명>
    cascade model loader
    1. seg_model_path : segmentation model path
    2. inpaint_model_path : inpainting model path
    
    3. device : device
    
    4. init_seg_model : segmentation model을 초기화하는 함수
    5. load_seg_model : segmentation model을 로드하는 함수
    6. init_inpaint_model : inpainting model을 초기화하는 함수
    
    7. load_inpaint_model : inpainting model을 로드하는 함수
    8. get_cascade_model_name : cascade model의 이름을 반환하는 함수
    9. load_models : segmentation model과 inpainting model을 로드하는 함수
    
    10. seg_model : segmentation model
    11. inpaint_model : inpainting model
    12. seg_model_name : segmentation model name
    13. inpaint_model_name : inpainting model name
    14. seg_model_path : segmentation model path
    15. inpaint_model_path : inpainting model path
    16. device : device
    
    17. cascade_model_name : cascade model name
    
    <주의>
    1. segmentation model과 inpainting model의 이름을 가져오기 위해서는 get_cascade_model_name 함수를 사용해야함 
    2. segmentation model과 inpainting model을 로드하기 위해서는 load_models 함수를 사용해야함 
    3. segmentation model과 inpainting model을 초기화하기 위해서는 init_seg_model, init_inpaint_model 함수를 사용해야함 
    4. segmentation model과 inpainting model을 로드하기 위해서는 load_seg_model, load_inpaint_model 함수를 사용해야함 
    
    """
    def __init__(self, seg_model_path, inpaint_model_path, device, width = 512, height = 512):
        self.seg_model_name = seg_model_path.split('/')[-2]
        self.inpaint_model_name = inpaint_model_path.split('/')[-2]
        self.seg_model_path = seg_model_path
        self.inpaint_model_path = inpaint_model_path
        self.device = device
        self.width, self.height = width, height 
        
    def init_seg_model(self):
        model_save_path = os.path.dirname(self.seg_model_path)
        model_version = self.seg_model_path.split('/')[-1]
        if self.seg_model_path.split('/')[-2].split('_')[0] == 'monai':
            model_name = 'monai_swinunet'
        else:
            model_name = self.seg_model_path.split('/')[-2].split('_')[0]
        print(f" Model save path : {model_save_path}")
        print(f" Model version : {model_version}")
        print(f" Model name : {model_name}")
        
        self.load_seg_model(model_save_path, model_version, model_name)
        
    def load_seg_model(self, model_save_path, model_version, model_name):
        checkpoint = torch.load(os.path.join(model_save_path, model_version), map_location= self.device)['model_state_dict']
        
        seg_model_loader = load_segment_model.segmentation_models_loader(
            model_name = model_name, width = self.width, height = self.height
        )
        self.seg_model = seg_model_loader.load_model().to(self.device)
        self.seg_model.load_state_dict(checkpoint)
    
    def init_inpaint_model(self):
        model_save_path = os.path.dirname(self.inpaint_model_path)
        model_version = self.inpaint_model_path.split('/')[-1]
        model_name = self.inpaint_model_path.split('/')[-2].split('_')[0]
        print(f" Model save path : {model_save_path}")
        print(f" Model version : {model_version}")
        print(f" Model name : {model_name}")
        
        self.load_inpaint_model(model_save_path, model_version, model_name)

    def load_inpaint_model(self, model_save_path, model_version, model_name):
        checkpoint = torch.load(os.path.join(model_save_path, model_version), map_location= self.device)['netG_state_dict']
        inpaint_model_loader = load_inpaint_model.inpainting_models_loader(
            model_name = model_name, width = self.width, height = self.height
        )
        self.inpaint_model = inpaint_model_loader.load_model().to(self.device)
        self.inpaint_model.load_state_dict(checkpoint)
    def get_cascade_model_name(self):
        cascade_model_name = self.seg_model_name + '@' + self.inpaint_model_name
        return cascade_model_name 
        
        
    def load_models(self):
        self.init_seg_model()
        self.init_inpaint_model()
        
        return self.seg_model, self.inpaint_model