#!/bin/sh

# Without affine transform, without DANR
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 /home/DANR/train_distributed_indoor.py \
--exp_id indoor_no_affine --arch resnest_50 --dataset indoor --num_classes 7 --batch_size 16 --gpus 0,1,2,3 \
--image_mixup --lr_step 90,110 --num_epochs 120 --save_all \
--not_rand_crop --scale 1.0 --rotate 0 --shift 0  # nullify affine transform


# Without affine transform, with DANR
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 /home/DANR/train_distributed_indoor_render512native.py \
 --exp_id indoor_render512native_split55_no_affine --arch resnest_50 --dataset indoor --num_classes 7 --batch_size 16 --gpus 0,1,2,3 \
 --lr_step 90,110 --num_epochs 120 --save_all \
 --not_rand_crop --scale 1.0 --rotate 0 --shift 0  # nullify affine transform


# validation & evaluation
python test_det_indoor.py \
--exp_id indoor_no_affine --arch resnest_50 --dataset indoor --num_classes 7 --gpus 0 --not_prefetch_test --center_thresh 0.2 \
--load_model /home/DANR/exp/ctdet/indoor_no_affine/model_best.pth

python test_det_indoor.py \
--exp_id indoor_render512native_split55_no_affine --arch resnest_50 --dataset indoor --num_classes 7 --gpus 0 --not_prefetch_test --center_thresh 0.2 \
--load_model /home/DANR/exp/ctdet/indoor_render512native_split55_no_affine/model_best.pth