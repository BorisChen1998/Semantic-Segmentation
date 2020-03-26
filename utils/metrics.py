import numpy as np
import torch

def compute_iou(pred, label, device, num_classes):
    pred = torch.argmax(pred, axis=0)
    #classes = torch.unique(label).to(device)
    num = torch.tensor([0 for i in range(num_classes-1)]).to(device)
    den = torch.tensor([0 for i in range(num_classes-1)]).to(device)
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


def compute_iou_batch(preds, labels, device, num_classes):
    nums = torch.tensor([0 for i in range(num_classes-1)]).to(device)
    dens = torch.tensor([0 for i in range(num_classes-1)]).to(device)
    for pred, label in zip(preds, labels):
        num, den = compute_iou(pred, label, device, num_classes)
        nums += num
        dens += den
    return (nums, dens)
