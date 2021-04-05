#!/bin/sh

# scarcity: 3/7, without DANR
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 /home/DANR/train_distributed_indoor.py \
--exp_id indoor_split37 --arch resnest_50 --dataset indoor --num_classes 7 --batch_size 16 --gpus 0,1,2,3 \
--lr_step 90,110 --num_epochs 120 --save_all \
--scarcity 37   # train/val split = 3/7


# scarcity: 3/7, with DANR
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 /home/DANR/train_distributed_indoor_render512pointSS.py \
--exp_id indoor_render_512pointSS_split37 --arch resnest_50 --dataset indoor --num_classes 7 --batch_size 16 --gpus 0,1,2,3 \
--lr_step 90,110 --num_epochs 120 --save_all \
--scarcity 37   # train/val split = 3/7


# validation & evaluation
python test_det_indoor.py \
--exp_id indoor_split37 --arch resnest_50 --dataset indoor --num_classes 7 --gpus 0 --not_prefetch_test --center_thresh 0.2 \
--load_model /home/DANR/exp/ctdet/indoor_split37/model_last.pth \
--scarcity 37   # train/val split = 3/7

python test_det_indoor.py \
--exp_id indoor_render_512pointSS_split37 --arch resnest_50 --dataset indoor --num_classes 7 --gpus 0 --not_prefetch_test --center_thresh 0.2 \
--load_model /home/DANR/exp/ctdet/indoor_render512pointSS_split37/model_last.pth \
--scarcity 37   # train/val split = 3/7
