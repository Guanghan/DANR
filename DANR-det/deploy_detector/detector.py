from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os, cv2
import numpy as np
import time
import torch
import torchvision.transforms as transforms

from deploy_detector.utility import create_model, load_model, get_affine_transform, centernet_decode, centernet_post_process, flip_tensor

class CenterNetDetector(object):
    def __init__(self, opt):
        if opt.gpus[0] >= 0:
            opt.device = torch.device('cuda')
        else:
            opt.device = torch.device('cpu')

        print('Creating model...')
        self.model = create_model(opt.num_layers)
        self.model = load_model(self.model, opt.model_path)
        self.model = self.model.to(opt.device)
        self.model.eval()

        self.mean = np.array(opt.mean, dtype=np.float32).reshape(1, 1, 3)
        self.std = np.array(opt.std, dtype=np.float32).reshape(1, 1, 3)
        self.max_per_image = opt.K
        self.num_classes = opt.num_classes
        self.scales = opt.test_scales
        self.opt = opt
        self.img_name = None
        self.transform_list = [transforms.Normalize(mean=opt.mean, std=opt.std)]
        self.mean_gpu = torch.from_numpy(self.mean).to(self.opt.device).reshape(3)
        self.std_gpu = torch.from_numpy(self.std).to(self.opt.device).reshape(3)


    def pre_process_org(self, image, scale, meta=None):
        height, width = image.shape[0:2]
        new_height = int(height * scale)
        new_width = int(width * scale)

        inp_height, inp_width = self.opt.input_h, self.opt.input_w
        c = np.array([new_width / 2., new_height / 2.], dtype=np.float32)
        s = max(height, width) * 1.0

        trans_input = get_affine_transform(c, s, 0, [inp_width, inp_height])
        resized_image = cv2.resize(image, (new_width, new_height))
        inp_image = cv2.warpAffine(resized_image, trans_input, (inp_width, inp_height), flags=cv2.INTER_LINEAR)
        inp_image = ((inp_image / 255. - self.mean) / self.std).astype(np.float32)

        images = inp_image.transpose(2, 0, 1).reshape(1, 3, inp_height, inp_width)
        images = torch.from_numpy(images)

        meta = {'c': c, 's': s,
                'out_height': inp_height // self.opt.down_ratio,
                'out_width': inp_width // self.opt.down_ratio}
        return images, meta


    def pre_process_resize(self, image, scale=1, meta=None):
        height, width = image.shape[0:2]
        inp_height, inp_width = self.opt.input_h, self.opt.input_w

        ts = time.time()
        resized_image = cv2.resize(image, (inp_width, inp_height))

        ts = time.time()
        images = torch.from_numpy(resized_image).float()

        c = np.array([width / 2., height / 2.], dtype=np.float32)
        s = max(height, width) * 1.0
        meta = {'c': c, 's': s,
                'out_height': inp_height // self.opt.down_ratio,
                'out_width': inp_width // self.opt.down_ratio}
        return images, meta


    def pre_process_affine(self, image, scale=1, meta=None):
        height, width = image.shape[0:2]
        inp_height, inp_width = self.opt.input_h, self.opt.input_w

        ts = time.time()
        c = np.array([width / 2., height / 2.], dtype=np.float32)
        s = max(height, width) * 1.0
        trans_input = get_affine_transform(c, s, 0, [inp_width, inp_height])
        inp_image = cv2.warpAffine(image, trans_input, (inp_width, inp_height), flags=cv2.INTER_LINEAR)

        ts = time.time()
        images = torch.from_numpy(inp_image).float()

        meta = {'c': c, 's': s,
                'out_height': inp_height // self.opt.down_ratio,
                'out_width': inp_width // self.opt.down_ratio}
        return images, meta


    def process(self, images, return_time=False):
        with torch.no_grad():
            output = self.model(images)[-1]
            hm = output['hm'].sigmoid_()
            wh = output['wh']
            reg = output['reg']
            if self.opt.flip_test:
                hm = (hm[0:1] + flip_tensor(hm[1:2])) / 2
                wh = (wh[0:1] + flip_tensor(wh[1:2])) / 2
                reg = reg[0:1]
            torch.cuda.synchronize()
            forward_time = time.time()
            dets = centernet_decode(hm, wh, reg=reg,
                                    cat_spec_wh=self.opt.cat_spec_wh, K=self.opt.K)

        if return_time:
            return output, dets, forward_time
        else:
            return output, dets

    def post_process(self, dets, meta, scale=1):
        dets = dets.detach().cpu().numpy()
        dets = dets.reshape(1, -1, dets.shape[2])
        dets = centernet_post_process(dets.copy(),
                                      [meta['c']], [meta['s']], meta['out_height'],
                                      meta['out_width'],
                                      self.opt.num_classes)
        for j in range(1, self.num_classes + 1):
            dets[0][j] = np.array(dets[0][j], dtype=np.float32).reshape(-1, 5)
            dets[0][j][:, :4] /= scale
        return dets[0]

    def merge_outputs(self, detections):
        results = {}
        for j in range(1, self.num_classes + 1):
            results[j] = np.concatenate(
                [detection[j] for detection in detections], axis=0).astype(np.float32)
        scores = np.hstack(
            [results[j][:, 4] for j in range(1, self.num_classes + 1)])
        if len(scores) > self.max_per_image:
            kth = len(scores) - self.max_per_image
            thresh = np.partition(scores, kth)[kth]
            for j in range(1, self.num_classes + 1):
                keep_inds = (results[j][:, 4] >= thresh)
                results[j] = results[j][keep_inds]
        return results

    def inference(self, image_or_path_or_tensor, meta=None):
        load_time, pre_time, net_time, dec_time, post_time = 0, 0, 0, 0, 0
        merge_time, tot_time = 0, 0
        start_time = time.time()
        pre_processed = False
        if isinstance(image_or_path_or_tensor, np.ndarray):
            image = image_or_path_or_tensor
        elif type(image_or_path_or_tensor) == type(''):
            image = cv2.imread(image_or_path_or_tensor)
            self.img_name = os.path.basename(image_or_path_or_tensor)
        else:
            image = image_or_path_or_tensor['image'][0].numpy()
            pre_processed_images = image_or_path_or_tensor
            pre_processed = True

        loaded_time = time.time()
        load_time += (loaded_time - start_time)

        detections = []
        for scale in self.scales:
            scale_start_time = time.time()
            if not pre_processed:
                images, meta = self.pre_process_affine(image, scale, meta)
                #images, meta = self.pre_process_resize(image, scale, meta)   # decreased performance
                #images, meta = self.pre_process_org(image, scale, meta)     # same performance with pre_process_affine
            else:
                # import pdb; pdb.set_trace()
                images = pre_processed_images['images'][scale][0]
                meta = pre_processed_images['meta'][scale]
                meta = {k: v.numpy()[0] for k, v in meta.items()}

            ts = time.time()
            images = images.to(self.opt.device)
            print("tensor to GPU time: {}".format(time.time() - ts))

            # if using pre_process_org, comment this code block
            #'''
            # perform image normalization on GPU
            ts = time.time()
            images = ((images / 255. - self.mean_gpu.expand_as(images)) / self.std_gpu.expand_as(images))
            images = images.permute((2, 0, 1))
            images = torch.unsqueeze(images, 0)
            print("GPU image normalization time: {}".format(time.time() - ts))
            #'''

            torch.cuda.synchronize()
            pre_process_time = time.time()
            pre_time += pre_process_time - scale_start_time

            output, dets, forward_time = self.process(images, return_time=True)

            torch.cuda.synchronize()
            net_time += forward_time - pre_process_time
            decode_time = time.time()
            dec_time += decode_time - forward_time

            dets = self.post_process(dets, meta, scale)
            torch.cuda.synchronize()
            post_process_time = time.time()
            post_time += post_process_time - decode_time

            detections.append(dets)

        results = self.merge_outputs(detections)
        torch.cuda.synchronize()
        end_time = time.time()
        merge_time += end_time - post_process_time
        tot_time += end_time - start_time

        return {'results': results, 'tot': tot_time, 'load': load_time,
                'pre': pre_time, 'net': net_time, 'dec': dec_time,
                'post': post_time, 'merge': merge_time}
