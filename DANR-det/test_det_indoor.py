from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import cv2
import numpy as np
from progress.bar import Bar
import torch

from opts import opts
from utils.misc import AverageMeter

from dataset.det.indoor_centernet import CenterIndoorDataset
from detectors.center_detector import CenterDetector

import os.path as osp
import sys

def add_path(path):
    if path not in sys.path:
        sys.path.insert(0, path)

this_dir = osp.dirname(__file__)

# Add lib to PYTHONPATH
lib_path = osp.join(this_dir, 'external')
add_path(lib_path)


class PrefetchDataset(torch.utils.data.Dataset):
    def __init__(self, opt, dataset, pre_process_func):
        self.images = dataset.images
        self.load_image_func = dataset.coco.loadImgs
        self.img_dir = dataset.img_dir
        self.pre_process_func = pre_process_func
        self.opt = opt

    def __getitem__(self, index):
        img_id = self.images[index]
        img_info = self.load_image_func(ids=[img_id])[0]
        img_path = os.path.join(self.img_dir, img_info['file_name'])
        image = cv2.imread(img_path)
        images, meta = {}, {}
        for scale in opt.test_scales:
            images[scale], meta[scale] = self.pre_process_func(image, scale)
        return img_id, {'images': images, 'image': image, 'meta': meta}

    def __len__(self):
        return len(self.images)


def prefetch_test(opt):
    os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpus_str

    opt = opts().update_dataset_info_and_set_heads(opt, CenterIndoorDataset)   # HERE!
    print(opt)

    split = 'val' if not opt.trainval else 'test'
    print("Split: {}".format(split))
    dataset = CenterIndoorDataset(opt, split)
    detector = CenterDetector(opt)

    data_loader = torch.utils.data.DataLoader(
        PrefetchDataset(opt, dataset, detector.pre_process),
        batch_size=1, shuffle=False, num_workers=1, pin_memory=True)

    results = {}
    num_iters = len(dataset)
    bar = Bar('{}'.format(opt.exp_id), max=num_iters)
    time_stats = ['tot', 'load', 'pre', 'net', 'dec', 'post', 'merge']
    avg_time_stats = {t: AverageMeter() for t in time_stats}
    for ind, (img_id, pre_processed_images) in enumerate(data_loader):
        ret = detector.run(pre_processed_images)
        results[img_id.numpy().astype(np.int32)[0]] = ret['results']
        Bar.suffix = '[{0}/{1}]|Tot: {total:} |ETA: {eta:} '.format(
            ind, num_iters, total=bar.elapsed_td, eta=bar.eta_td)
        for t in avg_time_stats:
            avg_time_stats[t].update(ret[t])
            Bar.suffix = Bar.suffix + '|{} {tm.val:.3f}s ({tm.avg:.3f}s) '.format(
                t, tm=avg_time_stats[t])
        bar.next()
    bar.finish()
    '''
    # evaluate bbox
    dataset.run_eval(results, opt.save_dir)
    # evaluate censorship
    dataset.run_censor_eval(results, opt.save_dir)
    '''
    # export qualitative results
    dataset.export_qualitative_results(results, split, thresh=0.15)


def test(opt):
    os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpus_str

    opt = opts().update_dataset_info_and_set_heads(opt, CenterIndoorDataset)   # HERE!
    print(opt)

    split = 'val' if not opt.trainval else 'test'
    print("Split: {}".format(split))
    dataset = CenterIndoorDataset(opt, split)
    detector = CenterDetector(opt)

    results = {}
    num_iters = len(dataset)
    bar = Bar('{}'.format(opt.exp_id), max=num_iters)
    time_stats = ['tot', 'load', 'pre', 'net', 'dec', 'post', 'merge']
    avg_time_stats = {t: AverageMeter() for t in time_stats}
    for ind in range(num_iters):
        img_id = dataset.images[ind]
        img_info = dataset.coco.loadImgs(ids=[img_id])[0]
        img_path = os.path.join(dataset.img_dir, img_info['file_name'])

        ret = detector.run(img_path)

        results[img_id] = ret['results']

        Bar.suffix = '[{0}/{1}]|Tot: {total:} |ETA: {eta:} '.format(
            ind, num_iters, total=bar.elapsed_td, eta=bar.eta_td)
        for t in avg_time_stats:
            avg_time_stats[t].update(ret[t])
            Bar.suffix = Bar.suffix + '|{} {:.3f} '.format(t, avg_time_stats[t].avg)
        bar.next()
    bar.finish()
    # evaluate bbox
    dataset.run_eval(results, opt.save_dir)
    '''
    # evaluate censorship
    dataset.run_censor_eval(results, opt.save_dir)
    # export qualitative results
    dataset.export_qualitative_results(results, split, thresh=0.15)
    '''


if __name__ == '__main__':
    opt = opts().parse()
    if opt.not_prefetch_test:
        test(opt)
    else:
        prefetch_test(opt)

