import torch
import torch.nn as nn
import torch.nn.functional as F

class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-6):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, predict, target):
        """
        predict: (Batch, Channels, D, H, W) - Model output (after Sigmoid)
        target: (Batch, Channels, D, H, W) - Ground truth mask
        """
        # Flatten the spatial dimensions (D*H*W) for each channel in the batch
        predict = predict.contiguous().view(predict.size(0), predict.size(1), -1)
        target = target.contiguous().view(target.size(0), target.size(1), -1)

        intersection = (predict * target).sum(dim=2)
        dice_score = (2. * intersection + self.smooth) / (
            predict.sum(dim=2) + target.sum(dim=2) + self.smooth
        )
        
        # Return 1 - Dice Score (we want to minimize the loss)
        return 1. - dice_score.mean()

class HybridLoss(nn.Module):
    """Combines Dice + BCE to leverage spatial focus and gradient stability."""
    def __init__(self, dice_weight=0.5, bce_weight=0.5):
        super(HybridLoss, self).__init__()
        self.dice = DiceLoss()
        self.dice_weight = dice_weight
        self.bce_weight = bce_weight
        
    def forward(self, predict, target):
        # Binary Cross-Entropy (BCE) for pixel-wise stability
        # target.float() is used because BCE expects floats
        bce_loss = F.binary_cross_entropy(predict, target.float())
        
        # Dice Loss for spatial overlap
        dice_loss = self.dice(predict, target)
        
        # Weighted sum of both losses
        return (self.bce_weight * bce_loss) + (self.dice_weight * dice_loss)