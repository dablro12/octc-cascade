#!/bin/bash

# Activate your virtual environment if needed
# source /home/eiden/anaconda3/bin/activate eiden
# Run the Python script with arguments
# --project_dir "/home/eiden/eiden/capstone/HUFS-BME-AI-WEB" \

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
python3 "$DIR/script.py" \
    --img_dir 'web/DB/data' \
    --save_dir 'web/DB/output' \
    --segment_model_path '/mnt/HDD/oci-seg_models/monai_swinunet_v4_240530/model_400.pt' \
    --inpaint_model_path '/mnt/HDD/oci_models/aotgan/OCI-GAN_v3_240508/model_64.pt' \
    --batch_size 1 \
    --seed 627 


    # --model_name "$1" \
