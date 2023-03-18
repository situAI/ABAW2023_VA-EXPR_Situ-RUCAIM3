import numpy as np
import torch
from sklearn.metrics import f1_score, accuracy_score

from utils.registery import METRIC_REGISTRY


def ccc_score(y_pred, y_true):
    """calc ccc score

    Args:
        y_pred (list(list(float))): 2-d pred value list
        y_true (list(list(float))): 2-d label value list

    Returns:
        ccc_score (float)
    """
    mean_true = np.mean(y_true)
    mean_pred = np.mean(y_pred)
    vx = y_true - mean_true
    vy = y_pred - mean_pred
    cor = np.sum(vx * vy) / (np.sqrt(np.sum(vx**2)) * np.sqrt(np.sum(vy**2)))

    sd_true = np.std(y_true)
    sd_pred = np.std(y_pred)

    numerator = 2 * cor * sd_true * sd_pred

    denominator = sd_true**2 + sd_pred**2 + (mean_true - mean_pred) ** 2

    return numerator / denominator


def one_hot_transfer(label, class_num):
    """transform to one hot encoding

    Args:
        label (list(int)): label list
        class_num (int): class number

    Returns:
        one-hot-style encoded array
    """
    return np.eye(class_num)[label]


@METRIC_REGISTRY.register()
class ExprMetric:
    def __call__(self, pred, gt, class_num=8):
        """calc expr metric

        Args:
            pred (list(list(int))): 2-d pred class list
            gt (list(list(int))): 2-d label class list
            class_num (int): expr class number (default: 8)

        Returns:
            F1_mean (float): macro expr F1 score
            acc (float): accuracy
            F1 (list(float)): each expr's F1 score
        """
        pred = pred.flatten().tolist()
        gt = gt.flatten().tolist()
        acc = accuracy_score(gt, pred)

        gt = one_hot_transfer(gt, class_num)
        pred = one_hot_transfer(pred, class_num)
        F1 = []
        for i in range(class_num):
            gt_ = gt[:, i]
            pred_ = pred[:, i]
            F1.append(f1_score(gt_, pred_))
        F1_mean = np.mean(F1)

        return {'F1': F1_mean, 'ACC': acc, 'F1_list': F1}


@METRIC_REGISTRY.register()
class VAMetric:
    def __call__(self, pred, gt):
        """calc va metric

        Args:
            pred (np.ndarray with size [seq_len * bs, 2]): 2-d pred valence list
            gt (np.ndarray with size [seq_len * bs, 2]): 2-d label valence list

        Returns:
            ccc_v (float): valence ccc score
            ccc_a (float): arousal ccc score
            final_metric (float): (ccc_v + ccc_a) * 0.5

        """
        pred_v = pred[:, 0].flatten().tolist()
        pred_a = pred[:, 1].flatten().tolist()

        gt_v = gt[:, 0].flatten().tolist()
        gt_a = gt[:, 1].flatten().tolist()

        ccc_v, ccc_a = ccc_score(pred_v, gt_v), ccc_score(pred_a, gt_a)

        return {'valence_ccc': ccc_v, 'arousal_ccc': ccc_a, 'va_metric': 0.5 * (ccc_v + ccc_a)}


