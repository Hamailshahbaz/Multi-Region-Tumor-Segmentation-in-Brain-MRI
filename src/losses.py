#src/losses.py
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for saving plots
import matplotlib.pyplot as plt

class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-6):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, predict, target):
        predict = predict.contiguous().view(predict.size(0), predict.size(1), -1)
        target = target.contiguous().view(target.size(0), target.size(1), -1)

        intersection = (predict * target).sum(dim=2)
        dice_score = (2. * intersection + self.smooth) / (
            predict.sum(dim=2) + target.sum(dim=2) + self.smooth
        )
        return 1. - dice_score.mean()


class HybridLoss(nn.Module):
    def __init__(self, dice_weight=0.5, bce_weight=0.5):
        super(HybridLoss, self).__init__()
        self.dice = DiceLoss()
        self.dice_weight = dice_weight
        self.bce_weight = bce_weight

    def forward(self, predict, target):
        bce_loss = F.binary_cross_entropy(predict, target.float())
        dice_loss = self.dice(predict, target)
        return (self.bce_weight * bce_loss) + (self.dice_weight * dice_loss)


def get_loss(loss_name: str):
    loss_name = loss_name.lower().strip()
    if loss_name in ["ce", "bce", "crossentropy"]:
        return nn.BCELoss()
    elif loss_name == "dice":
        return DiceLoss()
    elif loss_name in ["hybrid", "combined"]:
        return HybridLoss(dice_weight=0.5, bce_weight=0.5)
    else:
        raise ValueError(f"Unknown loss: '{loss_name}'. Choose from: 'ce', 'dice', 'hybrid'.")


def dice_coefficient_3d(prediction, target, smooth=1e-6):
    prediction = (prediction > 0.5).float()
    intersection = (prediction * target).sum(dim=(2, 3, 4))
    union = prediction.sum(dim=(2, 3, 4)) + target.sum(dim=(2, 3, 4))
    dice = (2. * intersection + smooth) / (union + smooth)
    return dice.mean()

def dice_per_class(prediction, target, smooth=1e-6):
    prediction = (prediction > 0.5).float()
    intersection = (prediction * target).sum(dim=(2, 3, 4))
    union = prediction.sum(dim=(2, 3, 4)) + target.sum(dim=(2, 3, 4))
    dice = (2. * intersection + smooth) / (union + smooth)
    return dice.mean(dim=0).cpu().numpy()