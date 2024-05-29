# python ./run_multi.py --model "AOTGAN-random-mask" \
#                 --version "v1" \
#                 --cuda "0"\
#                 --ts_batch_size 4\
#                 --vs_batch_size 2\
#                 --epochs 50\
#                 --loss "ce"\
#                 --optimizer "Adam"\
#                 --learning_rate 0.0001\
#                 --scheduler "lambda"\
#                 --pretrain "no" --pretrained_model "Places2" --error_signal no\
#                 --wandb "yes"\ > output.log 2>&1 &
python ./run.py --model "OCI-GAN" \
                --version "v3" \
                --save_path "/mnt/HDD/oci_models/aotgan" \
                --cuda "0"\
                --ts_batch_size 5\
                --vs_batch_size 2\
                --epochs 100\
                --loss "ce"\
                --optimizer "Adam"\
                --learning_rate 0.0001\
                --scheduler "lambda"\
                --pretrain "yes" --pretrained_model "premodel" --error_signal no\
                --wandb "yes"\ > output.log 2>&1 &


