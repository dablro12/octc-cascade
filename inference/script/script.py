import argparse 
from infer import Seg_Inpaint_Service
import os 
import sys 
_project_root = os.path.abspath('/home/eiden/eiden/octc-cascade')
sys.path.append(_project_root)

def parse_arguments():
    parser = argparse.ArgumentParser(description="Cascade Seg-Inpaint Service Arguments")
    
    parser.add_argument("--img_dir", type=str, default="web/DB/data", help="Image directory")
    parser.add_argument("--save_dir", type=str, default="web/DB/output", help="Save dir")
    parser.add_argument("--segment_model_path", type=str, default="/mnt/HDD/oci-seg_models/monai_swinunet_v4_240530/model_400.pt", help="Segmentation model path")
    parser.add_argument("--inpaint_model_path", type=str, default='/mnt/HDD/oci_models/aotgan/OCI-GAN_v3_240508/model_64.pt', help="Inpainting model path")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size")
    parser.add_argument("--seed", type=int, default=627, help="Random Seed")
    
    return parser.parse_args()
if __name__ == "__main__":
    args = parse_arguments()
    
    service = Seg_Inpaint_Service(
        img_dir = os.path.join(_project_root, args.img_dir),
        save_dir= os.path.join(_project_root, args.save_dir),
        segment_model_path= args.segment_model_path,
        inpaint_model_path= args.inpaint_model_path,
        batch_size= args.batch_size,
        seed = args.seed
    )
    
    service.run() 