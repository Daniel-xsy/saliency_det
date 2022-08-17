from turtle import forward
import numpy as np
import torch
import torch.nn as nn


def miss_detection_rate(output, target, thre=0.5):
    out_bin = output > thre
    gt_bin = target > thre
    mdr = torch.sum(gt_bin * ~out_bin) / max(1, torch.sum(gt_bin))
    return mdr


def false_alarm_rate(output, target, thre=0.5):
    out_bin = output > thre
    gt_bin = target > thre
    far = torch.sum(~gt_bin * out_bin) / max(1, torch.sum(out_bin))
    return far


class ConsistencyLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss = nn.MSELoss()
    def forward(self, feat1, feat2):
        return self.loss(feat1, feat2)
    

class DataLoss(nn.Module):
    def __init__(self, alpha=10):
        super().__init__()
        self.alpha = alpha
    def forward(self, fake1, fake2, gt):
        loss_1 = self.alpha * miss_detection_rate(fake1, gt) + false_alarm_rate(fake1, gt)
        loss_2 = miss_detection_rate(fake2, gt) + self.alpha * false_alarm_rate(fake2, gt)
        return loss_1 + loss_2
        


def assign_learning_rate(param_group, new_lr):
    param_group["lr"] = new_lr


def _warmup_lr(base_lr, warmup_length, step):
    return base_lr * (step + 1) / warmup_length


def cosine_lr(optimizer, base_lrs, warmup_length, steps):
    if not isinstance(base_lrs, list):
        base_lrs = [base_lrs for _ in optimizer.param_groups]
    assert len(base_lrs) == len(optimizer.param_groups)
    def _lr_adjuster(step):
        for param_group, base_lr in zip(optimizer.param_groups, base_lrs):
            if step < warmup_length:
                lr = _warmup_lr(base_lr, warmup_length, step)
            else:
                e = step - warmup_length
                es = steps - warmup_length
                lr = 0.5 * (1 + np.cos(np.pi * e / es)) * base_lr
            assign_learning_rate(param_group, lr)
    return _lr_adjuster


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val
        self.count += n
        self.avg = self.sum / self.count
        

def calculateF1Measure(output_image,gt_image,thre):
    output_image = torch.squeeze(output_image)
    gt_image = torch.squeeze(gt_image)

    out_bin = output_image > thre
    gt_bin = gt_image > thre

    recall = torch.sum(gt_bin * out_bin) / torch.max(1, torch.sum(gt_bin))
    prec   = torch.sum(gt_bin * out_bin) / torch.max(1, torch.sum(out_bin))
    F1 = 2 * recall * prec / torch.max(0.001, recall + prec)

    return F1