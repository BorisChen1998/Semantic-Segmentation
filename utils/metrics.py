import numpy as np
import torch

num_classes = 20


def compute_iou(pred, label):
    pred = torch.argmax(pred, axis=0)
    #classes = torch.unique(label).cuda()
    num = torch.tensor([0 for i in range(num_classes-1)]).cuda()
    den = torch.tensor([0 for i in range(num_classes-1)]).cuda()
    ignore = label == num_classes-1
    for c in range(num_classes-1):
        #if c == num_classes-1:
        #    continue
        pred_c = pred == c
        label_c = label == c
        intersection = (pred_c & label_c).sum() #torch.logical_and(pred_c, label_c).sum()
        union = (pred_c | label_c).sum() #torch.logical_or(pred_c, label_c).sum()
        ignore_part = (ignore & pred_c).sum()
        num[c] = intersection
        den[c] = union-ignore_part
    return (num, den)


def compute_iou_batch(preds, labels):
    nums = torch.tensor([0 for i in range(num_classes-1)]).cuda()
    dens = torch.tensor([0 for i in range(num_classes-1)]).cuda()
    for pred, label in zip(preds, labels):
        num, den = compute_iou(pred, label)
        nums += num
        dens += den
    return (nums, dens)
