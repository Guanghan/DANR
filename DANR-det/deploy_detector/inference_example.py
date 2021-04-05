from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import cv2
from deploy_detector.opt import opt
from deploy_detector.detector import CenterNetDetector
from deploy_detector.utility import censor, draw_bboxes_on_img, vis_censor


if __name__ == "__main__":
    os.environ['CUDA_VISIBLE_DEVICES'] = "0"

    # create detector
    detector = CenterNetDetector(opt)

    # perform object detection
    img_path = "test_img.jpg"
    img = cv2.imread(img_path)
    ret = detector.inference(img)   # or just call: ret = detector.inference(img_path)
    print("Detection results: {}".format(ret))

    '''
    Return a dictionary in the following format:
    {'results': results, 
     'tot': tot_time,     # total running time 
     'load': load_time,   # image loading time
     'pre': pre_time,     # image pre-processing time 
     'net': net_time,     # network inference time (GPU)
     'dec': dec_time,     # tensor decoding time (GPU)
     'post': post_time,   # post-processing time 
     'merge': merge_time} # merge test results from different scales and flipped image (can be turned off)
    '''

    # perform image censoring
    censor_pred, pred_pos_cat_list = censor(opt, ret['results'])
    print("The image should be censored: {}".format(censor_pred))

    # visualize bboxes
    img = draw_bboxes_on_img(img, ret['results'])

    # visualize censor info
    pred_censor_str = "YES" if censor_pred else "NO"
    pred_cls_str = ', '.join(pred_pos_cat_list)
    img = vis_censor(img,
                     gt_censor_str='unknown', gt_cls_str='unknown',
                     pred_censor_str=pred_censor_str, pred_cls_str=pred_cls_str,
                     extra_str="unknown")

    img_output_path = "output_img.jpg"
    cv2.imwrite(img_output_path, img)



