#!/bin/sh

# scarcity: 5/5, without DANR
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 /home/DANR/train_distributed_indoor.py \
--exp_id indoor --arch resnest_50 --dataset indoor --num_classes 7 --batch_size 16 --gpus 0,1,2,3 \
--image_mixup --lr_step 90,110 --num_epochs 120 --save_all


# scarcity: 5/5, with DANR
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 /home/DANR/train_distributed_indoor_render512native.py \
 --exp_id indoor_render512native_split55 --arch resnest_50 --dataset indoor --num_classes 7 --batch_size 16 --gpus 0,1,2,3 \
 --lr_step 90,110 --num_epochs 120 --save_all


# validation & evaluation
python test_det_indoor.py \
--exp_id indoor --arch resnest_50 --dataset indoor --num_classes 7 --gpus 0 --not_prefetch_test --center_thresh 0.2 \
--load_model /home/DANR/exp/ctdet/indoor/model_best.pth

python test_det_indoor.py \
--exp_id indoor_render512native_split55 --arch resnest_50 --dataset indoor --num_classes 7 --gpus 0 --not_prefetch_test --center_thresh 0.2 \
--load_model /home/DANR/exp/ctdet/indoor_render512native_split55/model_last.pth