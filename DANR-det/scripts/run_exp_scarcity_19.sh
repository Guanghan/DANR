#!/bin/sh

# scarcity: 1/9, without DANR
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 /home/DANR/train_distributed_indoor.py \
--exp_id indoor_split19 --arch resnest_50 --dataset indoor --num_classes 7 --batch_size 16 --gpus 0,1,2,3 \
--image_mixup --lr_step 90,110 --num_epochs 120 --save_all \
--scarcity 19   # train/val split = 1/9


# scarcity: 1/9, with DANR
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 /home/DANR/train_distributed_indoor_render512pointSS.py \
 --exp_id indoor_render_512pointSS_split19 --arch resnest_50 --dataset indoor --num_classes 7 --batch_size 16 --gpus 0,1,2,3 \
 --lr_step 90,110 --num_epochs 120 --save_all \
 --scarcity 19   # train/val split = 1/9


# validation & evaluation
python test_det_indoor.py \
--exp_id indoor_split19 --arch resnest_50 --dataset indoor --num_classes 7 --gpus 1 --not_prefetch_test --center_thresh 0.2 \
--load_model /home/DANR/exp/ctdet/indoor_split19/model_best.pth \
--scarcity 19   # train/val split = 1/9

python test_det_indoor.py \
--exp_id indoor_render_512pointSS_split19 --arch resnest_50 --dataset indoor --num_classes 7 --gpus 1 --not_prefetch_test --center_thresh 0.2 \
--load_model /home/DANR/exp/ctdet/indoor_render512pointSS_split19/model_best.pth \
--scarcity 19   # train/val split = 1/9
