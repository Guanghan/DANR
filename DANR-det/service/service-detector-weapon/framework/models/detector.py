from collections import OrderedDict
import numpy as np
import cv2
import torch
import torch.nn as nn
import time

from .model_resnest import get_pose_net


def get_dir(src_point, rot_rad):
    sn, cs = np.sin(rot_rad), np.cos(rot_rad)

    src_result = [0, 0]
    src_result[0] = src_point[0] * cs - src_point[1] * sn
    src_result[1] = src_point[0] * sn + src_point[1] * cs

    return src_result


def get_3rd_point(a, b):
    direct = a - b
    return b + np.array([-direct[1], direct[0]], dtype=np.float32)


def get_affine_transform(center,
                         scale,
                         rot,
                         output_size,
                         shift=np.array([0, 0], dtype=np.float32),
                         inv=0):
    if not isinstance(scale, np.ndarray) and not isinstance(scale, list):
        scale = np.array([scale, scale], dtype=np.float32)

    scale_tmp = scale
    src_w = scale_tmp[0]
    dst_w = output_size[0]
    dst_h = output_size[1]

    rot_rad = np.pi * rot / 180
    src_dir = get_dir([0, src_w * -0.5], rot_rad)
    dst_dir = np.array([0, dst_w * -0.5], np.float32)

    src = np.zeros((3, 2), dtype=np.float32)
    dst = np.zeros((3, 2), dtype=np.float32)
    src[0, :] = center + scale_tmp * shift
    src[1, :] = center + src_dir + scale_tmp * shift
    dst[0, :] = [dst_w * 0.5, dst_h * 0.5]
    dst[1, :] = np.array([dst_w * 0.5, dst_h * 0.5], np.float32) + dst_dir

    src[2:, :] = get_3rd_point(src[0, :], src[1, :])
    dst[2:, :] = get_3rd_point(dst[0, :], dst[1, :])

    if inv:
        trans = cv2.getAffineTransform(np.float32(dst), np.float32(src))
    else:
        trans = cv2.getAffineTransform(np.float32(src), np.float32(dst))

    return trans


class CenterNetDetector(object):

    def __init__(self, graph, num_classes=27):
        num_layers = 50
        heads = {'hm': num_classes,
                 'wh': 2,
                 'reg': 2,
                 }
        self.net = get_pose_net(num_layers, heads=heads, head_conv=64)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self._load_weight(graph)
        self.net.to(self.device)
        self.net.eval()
        self._warmup()


    def _warmup(self):
      image = np.random.rand(512, 512, 3)
      image = self._normalize(image)
      feat = self.inference(image)


    def _load_weight(self, graph):
      params = torch.load(graph, map_location='cpu')['state_dict']
      tmp_params = OrderedDict()
      for k,v in params.items():
        if 'module.' in k:
          tmp_params[k[7:]] = v
        else:
          tmp_params[k] = v
      params = tmp_params
      for n in self.net.state_dict().keys():
        if n in params and params[n].shape == self.net.state_dict()[n].shape:
          self.net.state_dict()[n][...] = params[n]
        else:
          print(n)


    def _normalize(self, in_img, mean=(0.408, 0.447, 0.470), variance=(0.289, 0.274, 0.278)):
        # should be BGR order
        img = np.array(in_img, copy=True).astype(np.float32)
        img -= np.array(mean, dtype=np.float32) * 255.
        img /= np.array(variance, dtype=np.float32) * 255.
        return img


    def preprocess_image(self, image, size):

        # Resize
        start_time = time.time()
        image = np.array(image)[:, :, ::-1]
        height, width = image.shape[0:2]
        inp_width, inp_height = size

        c = np.array([width / 2., height / 2.], dtype=np.float32)
        s = max(height, width) * 1.0

        trans_input = get_affine_transform(c, s, 0, [inp_width, inp_height])
        resized_image = cv2.warpAffine(image, trans_input, (inp_width, inp_height), flags=cv2.INTER_LINEAR)
        
        # Normalize
        resized_image = np.ascontiguousarray(resized_image)
        image = self._normalize(resized_image)

        # To tensor
        return image


    def inference(self, image):
        image = torch.from_numpy(image.transpose(2, 0, 1))
        image = image.unsqueeze_(0)
        image_tensor = image.to(self.device)
        with torch.no_grad():
          outputs = self.net(image_tensor)
          hm = outputs['hm'].sigmoid_()
          hm = hm[:,(2, 5, 7, 8, 11, 12, 13, 17, 19, 23, 27, 28)].reshape((hm.shape[0], -1))
          
        return hm.cpu().data.numpy().max(axis=1)
