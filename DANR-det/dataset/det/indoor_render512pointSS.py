from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pycocotools.coco as coco
from pycocotools.cocoeval import COCOeval
import numpy as np
import json
import os

import torch.utils.data as data


class IndoorRender512PointSS(data.Dataset):
    num_classes = 7
    default_resolution = [512, 512]

    mean = np.array([0.40789654, 0.44719302, 0.47026115],
                    dtype=np.float32).reshape(1, 1, 3)
    std = np.array([0.28863828, 0.27408164, 0.27809835],
                   dtype=np.float32).reshape(1, 1, 3)

    def __init__(self, opt, split):
        super(IndoorRender512PointSS, self).__init__()
        self.data_dir = os.path.join(opt.data_dir, 'indoor_render_512pointSS')
        self.img_dir = os.path.join(self.data_dir, 'images')

        if opt.scarcity == '55':
            # 50:50 split
            self.annot_path = os.path.join(self.data_dir, '{}.json'.format(split))
        elif opt.scarcity == '37':
            # 30:70 split
            self.annot_path = os.path.join(self.data_dir, '{}_37.json'.format(split))
        elif opt.scarcity == '19':
            # 10:90 split
            self.annot_path = os.path.join(self.data_dir, '{}_19.json'.format(split))

        self.max_objs = 128
        self.class_name = [
            '__background__',
            'chair',   # 1
            'clock',  # 2
            'exit', # 3
            'fireextinguisher',      # 4
            'printer',      # 5
            'screen',        # 6
            'trashbin']    # 7

        self.pos_ids = [1, 2, 3, 4, 5, 6, 7]  # remove knife(kitchen) from weapon
        self.neg_ids = [i for i in range(len(self.class_name)) if i not in self.pos_ids]
        self.ignore_cls_list = [] # do not train with these classes
        self.id_to_thresh = {}
        for pos_id in self.pos_ids:
            self.id_to_thresh[pos_id] = 0.2

        #self._valid_ids = [i for i in range(1, self.num_classes+1) if i not in self.ignore_cls_list]
        self._valid_ids = [i for i in range(1, self.num_classes+1)]
        self.cat_ids = {v: i for i, v in enumerate(self._valid_ids)}
        self.voc_color = [(v // 32 * 64 + 64, (v // 8) % 4 * 64, v % 8 * 32) \
                          for v in range(1, self.num_classes + 1)]
        self._data_rng = np.random.RandomState(123)
        self._eig_val = np.array([0.2141788, 0.01817699, 0.00341571],
                                 dtype=np.float32)
        self._eig_vec = np.array([
            [-0.58752847, -0.69563484, 0.41340352],
            [-0.5832747, 0.00994535, -0.81221408],
            [-0.56089297, 0.71832671, 0.41158938]
        ], dtype=np.float32)

        self.split = split
        self.opt = opt

        self.coco = coco.COCO(self.annot_path)
        self.images = self.coco.getImgIds()
        self.num_samples = len(self.images)

        print('Loaded {} {} samples'.format(split, self.num_samples))

    def _to_float(self, x):
        return float("{:.2f}".format(x))

    def convert_eval_format(self, all_bboxes):
        # import pdb; pdb.set_trace()
        detections = []
        for image_id in all_bboxes:
            for cls_ind in all_bboxes[image_id]:
                category_id = self._valid_ids[cls_ind - 1]
                for bbox in all_bboxes[image_id][cls_ind]:
                    bbox[2] -= bbox[0]
                    bbox[3] -= bbox[1]
                    score = bbox[4]
                    bbox_out = list(map(self._to_float, bbox[0:4]))

                    detection = {
                        "image_id": int(image_id),
                        "category_id": int(category_id),
                        "bbox": bbox_out,
                        "score": float("{:.2f}".format(score))
                    }
                    detections.append(detection)
        return detections

    def __len__(self):
        return self.num_samples

    def save_results(self, results, save_dir):
        json.dump(self.convert_eval_format(results),
                  open('{}/results.json'.format(save_dir), 'w'))

    def run_eval(self, results, save_dir):
        self.save_results(results, save_dir)
        coco_dets = self.coco.loadRes('{}/results.json'.format(save_dir))
        coco_eval = COCOeval(self.coco, coco_dets, "bbox")
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()


    def run_censor_eval(self, results, save_dir):
        self.save_results(results, save_dir)

        thresh_list = np.arange(0.05, 1.0, 0.05)
        prec_list = []
        recall_list = []
        true_pos_rate_list = [1.0]
        false_pos_rate_list = [1.0]
        for thresh in thresh_list:
            print("Working on thresh: {}".format(thresh))
            precision, recall, censor_rate, true_pos_rate, false_pos_rate = self.calculate_precision_recall_at_thresh(results, thresh)
            prec_list.append(precision)
            recall_list.append(recall)
            true_pos_rate_list.append(true_pos_rate)
            false_pos_rate_list.append(false_pos_rate)

        # draw precision-recall curve and calculate auc
        print("precision list: {}".format(prec_list))
        print("recall list: {}".format(recall_list))
        fig_save_path = os.path.join(self.opt.save_dir, "prec-recall-curve.png")
        self.plot_precision_recall_curve(fig_save_path, prec_list, recall_list)
        print("precision-recall-curve saved to {}".format(fig_save_path))

        # draw ROC-curve
        print("true_pos_rate_list: {}".format(true_pos_rate_list))
        print("false_pos_rate_list: {}".format(false_pos_rate_list))
        fig_save_path = os.path.join(self.opt.save_dir, "ROC-curve.png")
        self.plot_roc_curve(fig_save_path, true_pos_rate_list, false_pos_rate_list)
        print("ROC-curve saved to {}".format(fig_save_path))
        return


    def calculate_precision_recall_at_thresh(self, results, thresh):
        img_ids = self.coco.getImgIds()
        true_pos, true_neg, false_pos, false_neg = 0, 0, 0, 0

        num_imgs = len(img_ids)
        for img_id in img_ids:
            ''' Determine ground truth'''
            censor_gt = False
            ann_ids = self.coco.getAnnIds(imgIds=[img_id])
            anns = self.coco.loadAnns(ids=ann_ids)
            for ann in anns:
                cat_id = ann["category_id"]
                if cat_id in self.pos_ids:
                    censor_gt = True

            ''' Determine prediction '''
            censor_pred = False
            preds_all_cls = results[img_id]
            for cat_id in self.pos_ids:
                preds = preds_all_cls[cat_id]
                if preds == []: continue
                for pred in preds:
                    conf = pred[-1]
                    if conf > thresh:
                        censor_pred = True

            ''' Compare GT and Pred '''
            if censor_gt and censor_pred:
                true_pos += 1
            elif censor_gt and not censor_pred:
                false_neg += 1
            elif not censor_gt and not censor_pred:
                true_neg += 1
            elif not censor_gt and censor_pred:
                false_pos += 1

        ''' metrics '''
        precision = 1.0 * true_pos / (true_pos + false_pos) if true_pos + false_pos != 0 else 0
        recall = 1.0 * true_pos / (true_pos + false_neg) if true_pos + false_neg != 0 else 0
        censor_rate = 1.0 * (true_pos + false_pos) / num_imgs

        true_pos_rate = recall
        false_pos_rate = 1.0 * false_pos / (false_pos + true_neg) if false_pos + true_neg != 0 else 0 # fallout

        print("With thresh: {}".format(thresh))
        print("TP: {}, FP: {}, TN: {}, FN: {}".format(true_pos, false_pos, true_neg, false_neg))
        print("Precision: {}".format(precision))
        print("Recall / sensitivity: {}".format(recall))
        print("Censor rate: {}".format(censor_rate))
        print("True positive rate = recall")
        print("False positive rate: {}".format(false_pos_rate))
        return precision, recall, censor_rate, true_pos_rate, false_pos_rate


    def plot_precision_recall_curve(self, fig_save_path, prec_list, recall_list):
        import matplotlib.pyplot as plt
        from sklearn import metrics
        auc_score = metrics.auc(recall_list, prec_list)
        plt.clf()
        plt.plot(recall_list, prec_list)
        plt.text(0.5, 0.5, 'AUC(%s)'%(auc_score), fontsize=12)
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.ylim([0.0, 1.05])
        plt.xlim([0.0, 1.0])
        plt.title('Precision-Recall')
        plt.legend(loc="upper right")
        plt.savefig(fig_save_path)


    def plot_roc_curve(self, fig_save_path, true_pos_rate_list, false_pos_rate_list):
        import matplotlib.pyplot as plt
        from sklearn import metrics
        auc_score = metrics.auc(false_pos_rate_list, true_pos_rate_list)
        plt.clf()
        plt.plot(false_pos_rate_list, true_pos_rate_list)
        plt.text(0.5, 0.5, 'AUC(%s)'%(auc_score), fontsize=12)
        plt.xlabel('False positive rate')
        plt.ylabel('True positive rate')
        plt.ylim([0.0, 1.05])
        plt.xlim([0.0, 1.0])
        plt.title('ROC curve')
        plt.legend(loc="upper right")
        plt.savefig(fig_save_path)


    def batch_plot_precision_recall_curves(self, fig_save_path, curve_lists):
        '''
        :param curve_lists: a list of dictionaries, e.g., [dict_1, dict_2], where
                            dict_1 = { 'model_name': model_name,
                                       'prec_list': prec_list,
                                       'recall_list': recall_list}
        :return:
        '''
        import matplotlib.pyplot as plt
        from sklearn import metrics

        colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k']
        line_styles = ['-', '--', '-.', ':', '']
        plt.clf()
        fig, ax = plt.subplots()
        for i, curve in enumerate(curve_lists):
            recall_list = curve['recall_list']
            prec_list = curve['prec_list']
            model_name = curve['model_name']

            color = colors[i % len(colors)]
            line_style = line_styles[i%len(line_styles)]

            auc_score = metrics.auc(recall_list, prec_list)
            label_str = '{:s}(AUC): {:03.2f}'.format(model_name, auc_score)

            ax.plot(recall_list, prec_list, color=color, linestyle=line_style, linewidth=2.0, label=label_str)

        ax.set_xlabel('Recall')
        ax.set_ylabel('Precision')
        ax.set_ylim([0.0, 1.05])
        ax.set_xlim([0.0, 1.0])
        ax.set_title('Precision-Recall')
        legend = ax.legend(loc='upper right', shadow=True, fontsize='x-large')

        # Put a nicer background color on the legend.
        legend.get_frame().set_facecolor('C0')

        # save figure
        plt.savefig(fig_save_path)
        return


    def export_qualitative_results(self, results, split, thresh):
        from utils.utils_vis_bbox import draw_bboxes_on_img
        from utils.utils_vis_censor import vis_censor
        from utils.utils_io_folder import create_folder, remove_files_in_folder
        import cv2

        # clean previous results
        img_output_parent_folder = os.path.join(self.opt.save_dir, "qualitative", split)
        for extra_str in ["TP", "TN", "FP", "FN"]:
            img_output_folder_path = os.path.join(img_output_parent_folder, extra_str)
            remove_files_in_folder(img_output_folder_path)

        img_ids = self.coco.getImgIds()
        true_pos, true_neg, false_pos, false_neg = 0, 0, 0, 0

        num_imgs = len(img_ids)
        for img_id in img_ids:
            ''' Load image '''
            file_name = self.coco.loadImgs(ids=[img_id])[0]['file_name']
            print("filename = {}".format(file_name))
            img_path = os.path.join(self.img_dir, file_name)
            img = cv2.imread(img_path)

            ''' Determine ground truth'''
            censor_gt = False
            ann_ids = self.coco.getAnnIds(imgIds=[img_id])
            anns = self.coco.loadAnns(ids=ann_ids)
            gt_pos_cat_list = []
            for ann in anns:
                cat_id = ann["category_id"]
                if cat_id in self.pos_ids:
                    censor_gt = True
                    if self.class_name[cat_id] not in gt_pos_cat_list:
                        gt_pos_cat_list.append(self.class_name[cat_id])

            ''' Determine prediction '''
            censor_pred = False
            preds_all_cls = results[img_id]
            pred_pos_cat_list = []
            for cat_id in self.pos_ids:
                preds = preds_all_cls[cat_id]
                if preds == []: continue
                for pred in preds:
                    conf = pred[-1]
                    #if conf > self.id_to_thresh[cat_id]:   # using class-specific thresholds (during inference)
                    if conf > thresh:
                        censor_pred = True
                        if self.class_name[cat_id] not in pred_pos_cat_list:
                            pred_pos_cat_list.append(self.class_name[cat_id])

            ''' Compare GT and Pred '''
            if censor_gt and censor_pred:
                true_pos += 1
                extra_str= "TP"
            elif censor_gt and not censor_pred:
                false_neg += 1
                extra_str= "FN"
            elif not censor_gt and not censor_pred:
                true_neg += 1
                extra_str= "TN"
            elif not censor_gt and censor_pred:
                false_pos += 1
                extra_str= "FP"

            ''' visualize results for this image '''
            # add bboxes
            img = draw_bboxes_on_img(img, preds_all_cls)

            # add censor info
            gt_censor_str = "YES" if censor_gt else "NO"
            pred_censor_str= "YES" if censor_pred else "NO"
            gt_cls_str = ', '.join(gt_pos_cat_list)
            pred_cls_str = ', '.join(pred_pos_cat_list)
            img = vis_censor(img,
                             gt_censor_str, gt_cls_str,
                             pred_censor_str, pred_cls_str,
                             extra_str)

            img_output_folder = os.path.join(self.opt.save_dir, "qualitative", split, extra_str)
            create_folder(img_output_folder)
            img_output_path = os.path.join(img_output_folder, file_name)
            cv2.imwrite(img_output_path, img)


        ''' metrics '''
        precision = 1.0 * true_pos / (true_pos + false_pos)
        recall = 1.0 * true_pos / (true_pos + false_neg)
        censor_rate = 1.0 * (true_pos + false_pos) / num_imgs

        true_pos_rate = recall
        false_pos_rate = 1.0 * false_pos / (false_pos + true_neg) # fallout

        print("With thresh: {}".format(thresh))
        print("TP: {}, FP: {}, TN: {}, FN: {}".format(true_pos, false_pos, true_neg, false_neg))
        print("Precision: {}".format(precision))
        print("Recall / sensitivity: {}".format(recall))
        print("Censor rate: {}".format(censor_rate))
        print("True positive rate = recall")
        print("False positive rate: {}".format(false_pos_rate))
        return precision, recall, censor_rate, true_pos_rate, false_pos_rate
