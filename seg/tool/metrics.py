# import numpy as np


# class Evaluator(object):
#     def __init__(self, num_class, ignore=False):
#         self.num_class = num_class
#         self.ignore = ignore
#         self.confusion_matrix = np.zeros((self.num_class,)*2)

#     def Precision_Recall(self):
#         precision = np.diag(self.confusion_matrix) / (np.sum(self.confusion_matrix, axis=0) + 1e-5)
#         recall = np.diag(self.confusion_matrix) / (np.sum(self.confusion_matrix, axis=1) + 1e-5)
#         if self.ignore:
#             mp = np.nanmean(precision[:-1])
#             mr = np.nanmean(recall[:-1])
#             return precision, recall, mp, mr
#         else:
#             mp = np.nanmean(precision)
#             mr = np.nanmean(recall)
#             return precision, recall, mp, mr

#     def Pixel_Accuracy(self):
#         Acc = np.diag(self.confusion_matrix).sum() / self.confusion_matrix.sum()
#         return Acc

#     def Pixel_Accuracy_Class(self):
#         Acc = np.diag(self.confusion_matrix) / self.confusion_matrix.sum(axis=1)
#         if self.ignore:
#             Acc = np.nanmean(Acc[:-1])
#         else:
#             Acc = np.nanmean(Acc)
#         return Acc

#     def Mean_Intersection_over_Union(self):
#         IoU = np.diag(self.confusion_matrix) / (
#                     np.sum(self.confusion_matrix, axis=1) + np.sum(self.confusion_matrix, axis=0) -
#                     np.diag(self.confusion_matrix))
#         if self.ignore:
#             MIoU = np.nanmean(IoU[:-1])
#             return IoU[:-1], MIoU
#         else:
#             MIoU = np.nanmean(IoU)
#             return IoU, MIoU

#     def Frequency_Weighted_Intersection_over_Union(self):
#         freq = np.sum(self.confusion_matrix, axis=1) / np.sum(self.confusion_matrix)
#         iu = np.diag(self.confusion_matrix) / (
#                     np.sum(self.confusion_matrix, axis=1) + np.sum(self.confusion_matrix, axis=0) -
#                     np.diag(self.confusion_matrix))

#         FWIoU = (freq[freq > 0] * iu[freq > 0]).sum()
#         return FWIoU

#     def _generate_matrix(self, gt_image, pre_image):
#         mask = (gt_image >= 0) & (gt_image < self.num_class)
#         label = self.num_class * gt_image[mask].astype('int') + pre_image[mask]
#         count = np.bincount(label, minlength=self.num_class**2)
#         confusion_matrix = count.reshape(self.num_class, self.num_class)
#         return confusion_matrix

#     def add_batch(self, gt_image, pre_image):
#         assert gt_image.shape == pre_image.shape, "gt: {} pred: {}".format(gt_image.shape, pre_image.shape)
#         self.confusion_matrix += self._generate_matrix(gt_image, pre_image)

#     def reset(self):
#         self.confusion_matrix = np.zeros((self.num_class,) * 2)




import numpy as np
from medpy.metric.binary import hd95

class Evaluator(object):
    def __init__(self, num_class, ignore=False):
        self.num_class = num_class
        self.ignore = ignore
        self.confusion_matrix = np.zeros((self.num_class,)*2)
        # NEW: Accumulators for HD95
        self.hd95_sum = np.zeros(self.num_class)
        self.hd95_count = np.zeros(self.num_class)

    def Precision_Recall(self):
        precision = np.diag(self.confusion_matrix) / (np.sum(self.confusion_matrix, axis=0) + 1e-5)
        recall = np.diag(self.confusion_matrix) / (np.sum(self.confusion_matrix, axis=1) + 1e-5)
        if self.ignore:
            mp = np.nanmean(precision[:-1])
            mr = np.nanmean(recall[:-1])
            return precision, recall, mp, mr
        else:
            mp = np.nanmean(precision)
            mr = np.nanmean(recall)
            return precision, recall, mp, mr

    def Pixel_Accuracy(self):
        Acc = np.diag(self.confusion_matrix).sum() / self.confusion_matrix.sum()
        return Acc

    def Pixel_Accuracy_Class(self):
        Acc = np.diag(self.confusion_matrix) / self.confusion_matrix.sum(axis=1)
        if self.ignore:
            Acc = np.nanmean(Acc[:-1])
        else:
            Acc = np.nanmean(Acc)
        return Acc

    def Mean_Intersection_over_Union(self):
        IoU = np.diag(self.confusion_matrix) / (
                    np.sum(self.confusion_matrix, axis=1) + np.sum(self.confusion_matrix, axis=0) -
                    np.diag(self.confusion_matrix))
        if self.ignore:
            MIoU = np.nanmean(IoU[:-1])
            return IoU[:-1], MIoU
        else:
            MIoU = np.nanmean(IoU)
            return IoU, MIoU

    def Mean_Dice(self):
        Dice = 2 * np.diag(self.confusion_matrix) / (
                    np.sum(self.confusion_matrix, axis=1) + np.sum(self.confusion_matrix, axis=0) + 1e-10)
        if self.ignore:
            MDice = np.nanmean(Dice[:-1])
            return Dice[:-1], MDice
        else:
            MDice = np.nanmean(Dice)
            return Dice, MDice

    def Mean_HD95(self):
        HD95 = np.zeros(self.num_class)
        for k in range(self.num_class):
            if self.hd95_count[k] > 0:
                HD95[k] = self.hd95_sum[k] / self.hd95_count[k]
            else:
                HD95[k] = 0.0

        if self.ignore:
            valid_hd95 = [HD95[k] for k in range(self.num_class - 1) if self.hd95_count[k] > 0]
            MHD95 = np.mean(valid_hd95) if len(valid_hd95) > 0 else 0.0
            return HD95[:-1], MHD95
        else:
            valid_hd95 = [HD95[k] for k in range(self.num_class) if self.hd95_count[k] > 0]
            MHD95 = np.mean(valid_hd95) if len(valid_hd95) > 0 else 0.0
            return HD95, MHD95

    def Frequency_Weighted_Intersection_over_Union(self):
        freq = np.sum(self.confusion_matrix, axis=1) / np.sum(self.confusion_matrix)
        iu = np.diag(self.confusion_matrix) / (
                    np.sum(self.confusion_matrix, axis=1) + np.sum(self.confusion_matrix, axis=0) -
                    np.diag(self.confusion_matrix))

        FWIoU = (freq[freq > 0] * iu[freq > 0]).sum()
        return FWIoU

    def _generate_matrix(self, gt_image, pre_image):
        mask = (gt_image >= 0) & (gt_image < self.num_class)
        label = self.num_class * gt_image[mask].astype('int') + pre_image[mask]
        count = np.bincount(label, minlength=self.num_class**2)
        confusion_matrix = count.reshape(self.num_class, self.num_class)
        return confusion_matrix

    def add_batch(self, gt_image, pre_image):
        assert gt_image.shape == pre_image.shape, "gt: {} pred: {}".format(gt_image.shape, pre_image.shape)
        self.confusion_matrix += self._generate_matrix(gt_image, pre_image)

        cal = (gt_image < 255) # ignore VOC void class 255
        for k in range(self.num_class):
            gt_k = (gt_image == k) * cal
            pred_k = (pre_image == k) * cal
            sum_gt = np.sum(gt_k)
            sum_pred = np.sum(pred_k)
            
            if sum_gt > 0 or sum_pred > 0:
                if sum_gt == 0 or sum_pred == 0:
                    h_val = 373.12866
                else:
                    try:
                        h_val = hd95(pred_k, gt_k, voxelspacing=(1, 1))
                    except:
                        h_val = 373.12866
                self.hd95_sum[k] += h_val
                self.hd95_count[k] += 1

    def reset(self):
        self.confusion_matrix = np.zeros((self.num_class,) * 2)
        self.hd95_sum = np.zeros(self.num_class)
        self.hd95_count = np.zeros(self.num_class)