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
python ./VAE_run.py --model "VAE" \
                --version "v1" \
                --save_path "/mnt/HDD/oci_models/models" \
                --cuda "0"\
                --ts_batch_size 5\
                --vs_batch_size 2\
                --epochs 100\
                --loss "ce"\
                --optimizer "Adam"\
                --learning_rate 0.0002\
                --scheduler "lambda"\
                --pretrain "no" --pretrained_model "premodel" --error_signal no\
                --wandb "yes"\ > output.log 2>&1 &


