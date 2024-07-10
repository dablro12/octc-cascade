#!/bin/bash

# $1: model_name

/home/eiden/anaconda3/envs/chest-segmentation/bin/python3 "/home/eiden/eiden/octc-cascade/inference/script/script.py" \
    --img_dir '/home/eiden/eiden/octc-cascade/web/DB/data' \
    --save_dir '/home/eiden/eiden/octc-cascade/web/DB/output' \
    --segment_model_path '/mnt/HDD/oci-seg_models/monai_swinunet_v4_240530/model_400.pt' \
    --inpaint_model_path '/mnt/HDD/oci_models/aotgan/OCI-GAN_v3_240508/model_64.pt' \
    --batch_size 1 \
    --seed 627 


    # --model_name "$1" \
