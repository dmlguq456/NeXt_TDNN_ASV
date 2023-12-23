#! /usr/bin/python
# -*- encoding: utf-8 -*-

import torch
import torch.nn.functional as F
import numpy as np
from sklearn import metrics
from scipy.optimize import brentq
from scipy.interpolate import interp1d


# ================================
# for evaluation(@ speaker identification)
# ================================
def accuracy(output, target, topk=(1,)):
    """
    Computes the precision@k for the specified values of k
    Ref: https://github.com/clovaai/voxceleb_trainer/blob/master/utils.py
    """
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


# ================================
# for evaluation(@ verification)
# ================================
def compute_eer(scores, labels):
    """
    compute eer and threshold of eer
    Ref1: https://github.com/clovaai/voxceleb_trainer/blob/master/tuneThreshold.py
    Ref2: https://github.com/seongmin-kye/meta-SR/blob/b4c1ea1728e33f7bbf7015c38f508f24594f3f88/EER_full.py#L125
    """
    # Ref: https://github.com/clovaai/voxceleb_trainer/blob/master/tuneThreshold.py
    fpr, tpr, threshold = metrics.roc_curve(labels, scores, pos_label=1)
    fnr = 1 - tpr

    idx_eer = np.nanargmin(np.absolute((fnr - fpr)))
    eer = max(fpr[idx_eer], fnr[idx_eer])*100

    eer_threshold = threshold[idx_eer]

    return eer, eer_threshold


def compute_MinDCF(scores, labels, p_target=0.01, c_miss=1, c_fa=1):            
    """MinDCF
    - ref: https://github.com/zyzisyz/mfa_conformer/blob/master/score/utils.py
    Computes the minimum of the detection cost function.  The comments refer to
    equations in Section 3 of the NIST 2016 Speaker Recognition Evaluation Plan.
    """
    fpr, tpr, thresholds = metrics.roc_curve(labels, scores, pos_label=1)
    fnr = 1.0 - tpr

    min_c_det = float("inf")
    min_c_det_threshold = thresholds[0]
    for i in range(0, len(fnr)):
        c_det = c_miss * fnr[i] * p_target + c_fa * fpr[i] * (1 - p_target)
        if c_det < min_c_det:
            min_c_det = c_det
            min_c_det_threshold = thresholds[i]
    c_def = min(c_miss * p_target, c_fa * (1 - p_target))
    min_dcf = min_c_det / c_def

    return min_dcf, min_c_det_threshold
