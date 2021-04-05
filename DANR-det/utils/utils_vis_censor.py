import os, cv2
import numpy as np

def vis_censor(img, gt_censor_str = "Yes", gt_cls_str="weapon, knife",
                    pred_censor_str = "Yes", pred_cls_str = "weapon, knife",
                    extra_str = "TP"):
    ht, wid, nchannels = img.shape

    # create extra img
    debug_ht = 512
    debug_wid = 500
    debug_img = np.ones((debug_ht,debug_wid,3), np.uint8)

    # resize img
    new_ht = debug_ht
    resize_ratio = 1.0 * ht / new_ht  # = wid / new_wid
    new_wid = int(1.0 * wid / resize_ratio)
    dim = (new_wid, new_ht)
    resized_img = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)

    # put text on extra img
    # font
    font = cv2.FONT_HERSHEY_SIMPLEX
    # org
    org = [10, 20]
    # fontScale
    fontScale = 0.51
    # Blue color in BGR
    color = (255, 255, 255)
    # Line thickness of 2 px
    thickness = 1

    # GT INFO
    text = 'GT INFO:'
    debug_img = cv2.putText(debug_img, text, tuple(org), font, fontScale, color, thickness, cv2.LINE_AA)
    org[1] += 20
    text = '   censor: {}'.format(gt_censor_str)
    debug_img = cv2.putText(debug_img, text, tuple(org), font, fontScale, color, thickness, cv2.LINE_AA)
    org[1] += 20
    text = '   classes to censor: {}'.format(gt_cls_str)
    debug_img = cv2.putText(debug_img, text, tuple(org), font, fontScale, color, thickness, cv2.LINE_AA)
    org[1] += 40

    # pred INFO
    text = 'Pred INFO:'
    debug_img = cv2.putText(debug_img, text, tuple(org), font, fontScale, color, thickness, cv2.LINE_AA)
    org[1] += 20
    text = '   censor: {}'.format(pred_censor_str)
    debug_img = cv2.putText(debug_img, text, tuple(org), font, fontScale, color, thickness, cv2.LINE_AA)
    org[1] += 20
    text = '   classes to censor: {}'.format(pred_cls_str)
    debug_img = cv2.putText(debug_img, text, tuple(org), font, fontScale, color, thickness, cv2.LINE_AA)
    org[1] += 40

    # extra INFO
    text = 'Extra INFO:'
    debug_img = cv2.putText(debug_img, text, tuple(org), font, fontScale, color, thickness, cv2.LINE_AA)
    org[1] += 20
    text = '   result: {}'.format(extra_str)
    debug_img = cv2.putText(debug_img, text, tuple(org), font, fontScale, color, thickness, cv2.LINE_AA)

    horizontalAppendedImg = np.hstack((resized_img, debug_img))
    #cv2.imshow('Horizontal Appended', horizontalAppendedImg)
    #cv2.waitKey(2000)
    #cv2.destroyAllWindows()
    return horizontalAppendedImg


