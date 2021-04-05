DANR
=======
This repository contains code to reproduce results of the following paper:

**Data Augmentation for Object Detection via Differentiable Neural Rendering**


----
## Overview

If a detection model is to be inevitably trained with limited data, 
one way to remedy this is switching from the default supervised learning paradigm to semi-supervised learning and self-supervised learning.
To seamlessly integrate and enhance existing supervised object detection methods, 
in this work, we focus on addressing the data scarcity problem from a fundamental viewpoint without changing the supervised learning paradigm. 
We propose a new offline data augmentation method for object detection, 
which semantically interpolates the training data with novel views.

Extensive experiments show that our method, as a cost-free tool to enrich images and labels, 
can significantly boost the performance of object detection systems with scarce training data.


**In this repository, we provide scripts to run all the experiments in the ablation studies.**

----
## Prerequisites
- Python 3.6.9
- Pytorch 1.5.0+cu92 
- TorchVision 0.6.0+cu92

**To minimize the effort in replicating the environment, 
we provide a pre-built docker image with which the code can be run seamlessly.**

```
docker pull danr2021/danr
```
```
docker run -it --rm --gpus 4 --shm-size=8g danr:latest
```

All of the following experiments are conducted on 4 V100 GPUs 
in distributed training mode with synchronized BatchNorm.
Single-GPU training may render slightly inferior performance.
	
----
## Getting Started

### 1. Datasets

The datasets needed to run the experiments are already included in the docker image.

### 2. Experiments


**(1). Ablation on compatibility**:

We first check the compatibility of the proposed DANR with online augmentation methods.

We add a keypoint-based image mixup strategy to represent online augmentation, denoted as Online. 
We add the data generated offline by the DANR to training, denoted as Offline. 
In all of the other experiments, we only compare +Online vs. +Online+Offline, denoted as N.A. and +Aug, respectively.

Results show that DANR is compatible with online mixup. 
Their augmentation strategies are complementary. 
DANR alone can boost the performance by around 10 percent.

To reproduce experiment (1)：

- baseline (N.A.) vs. offline augmentation DANR:

	```
	bash ./scripts/run_exp_comp_baseline.sh
	```
	```
	bash ./scripts/run_exp_comp_offline.sh
	```
- baseline (N.A.) vs. online augmentation keypoint-based image mix-up:
	```
	bash ./scripts/run_exp_comp_online.sh
	```
- baseline (N.A.) vs. online & offline augmentation:
	```
	bash ./scripts/run_exp_comp_both.sh
	```

**(2). Ablation on render resolutions**:

With DANR, the detection performance is significantly improved. 
However, we notice that the average precision for small objects suffer a performance drop. 
We hypothesize that it is due to the small render resolution (e.g. 256 × 256) compared to the detector input resolution (e.g. 512 × 512).

The results show that, 
(a) augmentation with images rendered at different resolutions consistently boosts the 
detection performance;
(b) synthesized images at low resolutions may potentially lose some details compared to 
real images, which does harm to the detection performance of very small objects; 
(c) uplifting the image resolution via super-sampling further improves the performance; 
(d) super-sampling from points may be a better choice than super-sampling from pixels; 
(e) training a network with dense features and points achieves the best performance.


To reproduce experiment (2)：

- Render_256native:

	```
	bash ./scripts/run_exp_res_256native.sh
	```

- Render_512native:
	```
	bash ./scripts/run_exp_res_512native.sh
	```
	
- Render_512pixelSS:
	```
	bash ./scripts/run_exp_res_512pixelSS.sh
	```
	
- Render_512pointSS:
	```
	bash ./scripts/run_exp_res_512pointSS.sh
	```

**(3). Ablation on 3D and 2D semantics: DANR vs. Affine Transform**:

Considering the compatibility observed in experiment (1), we further experiment 
whether DANR is compatible with affine transform.

The extreme case of DANR is, when depth for each pixel is identical, 
the augmentation is equivalent to affine transform, which is linear.
This experiment is aimed to find out, whether the realistic 3D semantics captured by DANR will help the trainer boost the detection performance 
on top of the (potentially distorted) 2D semantics augmented by affine transform.
To study this, we deprive affine transform from the default baselines.

Results show that,
(1) DANR boosts object detection performance on top of affine transform;
(2) DANR improves performance with or without affine transform; 
(3) to some extent, DANR can make up for the loss of affine transform during training.

To reproduce experiment (3)：

- w/ affine vs. w/o affine:

	```
	bash ./scripts/run_exp_affine_w.sh
	```
	```
	bash ./scripts/run_exp_affine_wo.sh
	```
	
**(4): Ablation on scarcity (train/val split), 5/5 vs. 3/7 vs. 1/9**:

To study the best context of usage for DANR, 
we further experiment with three data-split settings: 5:5, 3:7 and 1:9, 
corresponding to different scarcity levels.
Highest degree of scarcity is 1:9.
While splitting images to train and validation sets, 
an image is assigned along with its corresponding augmented images.

The results show that, the scarcity of training data is more severe, 
the improvement achieved by our method is larger.

To reproduce experiment (4)：
- 5/5: 

	```
	bash ./scripts/run_exp_scarcity_55.sh
	```
- 3/7: 
	```
	bash ./scripts/run_exp_scarcity_37.sh
	```
- 1/9: 
	```
	bash ./scripts/run_exp_scarcity_19.sh
	```
	
**(5). Ablation on detectors**

Results show that, 
the DANR consistently boost the detection performance, 
regardless of the backbones used by the detector.

To reproduce experiment (5)：

- ResNet vs. ReNeSt:

	```
	bash ./scripts/run_exp_det_resnet.sh
	```
	```
	bash ./scripts/run_exp_det_resnest.sh
	```

## Citation
If you find this work helpful, please cite:
```
@article{ning2021data,
  title={Data Augmentation for Object Detection via Differentiable Neural Rendering},
  author={Ning, Guanghan and Chen, Guang and Tan, Chaowei and Luo, Si and Bo, Liefeng and Huang, Heng},
  journal={arXiv preprint arXiv:2103.02852},
  year={2021}
}
```