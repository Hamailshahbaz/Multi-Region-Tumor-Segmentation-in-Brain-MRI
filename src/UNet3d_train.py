import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for saving plots
import matplotlib.pyplot as plt

from data_loader import get_dataloader
from model import AttentionUNet25D, UNet25D

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


def dice_coefficient_2d(prediction, target, smooth=1e-6):
    prediction = (prediction > 0.5).float()
    intersection = (prediction * target).sum(dim=(2, 3))
    union = prediction.sum(dim=(2, 3)) + target.sum(dim=(2, 3))
    dice = (2. * intersection + smooth) / (union + smooth)
    return dice.mean()


def dice_per_class(prediction, target, smooth=1e-6):
    """Returns Dice for each class separately: [WT, TC, ET]"""
    prediction = (prediction > 0.5).float()
    intersection = (prediction * target).sum(dim=(2, 3))
    union = prediction.sum(dim=(2, 3)) + target.sum(dim=(2, 3))
    dice = (2. * intersection + smooth) / (union + smooth)
    return dice.mean(dim=0).cpu().numpy()  # (3,)

def visualize_predictions(images, masks, outputs, epoch, loss_name, model_name, out_dir="results/plots"):
    """
    Saves a figure with: input (first 4 channels), ground truth, and prediction.
    images:  (B, 12, H, W)
    masks:   (B, 3, H, W)
    outputs: (B, 3, H, W)
    """
    os.makedirs(out_dir, exist_ok=True)
    
    # Take first sample from batch
    img = images[0].cpu().numpy()   # (12, H, W)
    mask = masks[0].cpu().numpy()   # (3, H, W)
    pred = outputs[0].cpu().detach().numpy()  # (3, H, W)
    pred_bin = (pred > 0.5).astype(np.float32)

    fig, axes = plt.subplots(3, 3, figsize=(10, 10))
    class_names = ["Whole Tumor (WT)", "Tumor Core (TC)", "Enhancing Tumor (ET)"]

    for i in range(3):
        # Show one input channel (e.g., FLAIR center slice = channel 1)
        axes[i, 0].imshow(img[1], cmap='gray')
        axes[i, 0].set_title(f"Input (FLAIR) - {class_names[i]}")
        axes[i, 0].axis('off')

        axes[i, 1].imshow(mask[i], cmap='jet', vmin=0, vmax=1)
        axes[i, 1].set_title("Ground Truth")
        axes[i, 1].axis('off')

        axes[i, 2].imshow(pred_bin[i], cmap='jet', vmin=0, vmax=1)
        axes[i, 2].set_title(f"Prediction (Dice-based)")
        axes[i, 2].axis('off')

    plt.tight_layout()
    save_path = os.path.join(out_dir, f"{loss_name}_{model_name}_epoch{epoch:03d}.png")
    plt.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"  [Vis] Saved prediction plot: {save_path}")


def train_model(loss_name="hybrid", model_name="attention_unet", viz_every=5):
    # 1. Configuration
    data_dir = '/kaggle/input/datasets/awsaf49/brats20-dataset-training-validation/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_epochs = 25
    batch_size = 16
    learning_rate = 1e-4

    # Create output directories
    os.makedirs("results/models", exist_ok=True)
    os.makedirs("results/plots", exist_ok=True)

    # 2. DataLoaders (same as repo)
    train_loader, val_loader = get_dataloader(data_dir, batch_size=batch_size)

    # 3. Model
    if model_name.lower() == "attention_unet":
        model = AttentionUNet25D(in_channels=12, out_channels=3).to(device)
    elif model_name.lower() == "unet":
        model = UNet25D(in_channels=12, out_channels=3).to(device)
    else:
        raise ValueError(f"Unknown model: {model_name}")

    # 4. Loss & Optimizer
    criterion = get_loss(loss_name)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Track best model
    best_val_dice = 0.0
    best_epoch = 0

    print(f"Running {loss_name.upper()} Experiment on {device}...")
    print(f"Model: {model_name} | Loss: {criterion.__class__.__name__}")
    print("-" * 60)

    for epoch in range(1, num_epochs + 1):
        model.train()
        train_losses, train_dices = [], []

        for images, masks in train_loader:
            images, masks = images.to(device), masks.to(device)

            optimizer.zero_grad()
            outputs = model(images)

            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()

            train_losses.append(loss.item())
            train_dices.append(dice_coefficient_2d(outputs, masks).item())

        model.eval()
        val_losses, val_dices = [], []
        viz_images, viz_masks, viz_outputs = None, None, None

        with torch.no_grad():
            for batch_idx, (images, masks) in enumerate(val_loader):
                images, masks = images.to(device), masks.to(device)
                outputs = model(images)

                loss = criterion(outputs, masks)
                val_losses.append(loss.item())
                val_dices.append(dice_coefficient_2d(outputs, masks).item())

                # Store first batch for visualization
                if batch_idx == 0:
                    viz_images = images
                    viz_masks = masks
                    viz_outputs = outputs

        # Compute averages
        avg_train_loss = np.mean(train_losses)
        avg_train_dice = np.mean(train_dices)
        avg_val_loss = np.mean(val_losses)
        avg_val_dice = np.mean(val_dices)

        # Per-class Dice on last validation batch
        per_class_dice = dice_per_class(viz_outputs, viz_masks)

        print(f"Epoch {epoch}/{num_epochs}")
        print(f"  TRAIN | Loss: {avg_train_loss:.4f} | Dice: {avg_train_dice:.4f}")
        print(f"  VAL   | Loss: {avg_val_loss:.4f} | Dice: {avg_val_dice:.4f} "
              f"(WT={per_class_dice[0]:.3f}, TC={per_class_dice[1]:.3f}, ET={per_class_dice[2]:.3f})")

        if epoch % viz_every == 0 or epoch == 1:
            visualize_predictions(viz_images, viz_masks, viz_outputs, epoch, loss_name, model_name)

        if avg_val_dice > best_val_dice:
            best_val_dice = avg_val_dice
            best_epoch = epoch

            save_path = f"results/models/best_{loss_name}_{model_name}_25d.pth"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_dice': best_val_dice,
                'per_class_dice': per_class_dice,
            }, save_path)
            print(f"  [Best] New best model! Val Dice: {best_val_dice:.4f} (saved)")

        print("-" * 60)

    print(f"\nTraining complete. Best Val Dice: {best_val_dice:.4f} at Epoch {best_epoch}")
    print(f"Best model saved at: results/models/best_{loss_name}_{model_name}_25d.pth")


if __name__ == "__main__":
    for loss in ["ce"]:
        print(f"\n{'='*60}")
        print(f" STARTING EXPERIMENT: {loss.upper()} LOSS ")
        print(f"{'='*60}")
        train_model(loss_name=loss, model_name="attention_unet", viz_every=5)