#src/train.py
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from data_loader import get_dataloader_3d
from model import AttentionUNet3D, UNet3D
from losses import get_loss, dice_per_class, dice_coefficient_3d
import os
from visualize import visualize_predictions
from losses import get_loss, dice_per_class, dice_coefficient_3d, hd95_per_class_3d

def train_model(loss_name="hybrid", model_name="attention_unet", viz_every=5):
    # 1. Configuration
    data_dir = '/data/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_epochs = 25
    batch_size = 1
    learning_rate = 1e-4

    # Create output directories
    os.makedirs("results/models", exist_ok=True)
    os.makedirs("results/plots", exist_ok=True)

    # 2. DataLoaders (same as repo)
    train_loader, val_loader = get_dataloader_3d(data_dir, batch_size=batch_size)

    # 3. Model
    if model_name.lower() == "attention_unet":
        model = AttentionUNet3D(in_channels=4, out_channels=3).to(device)
    elif model_name.lower() == "unet":
        model = UNet3D(in_channels=4, out_channels=3).to(device)
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
            train_dices.append(dice_coefficient_3d(outputs, masks).item())

        model.eval()
        val_losses, val_dices = [], []
        viz_images, viz_masks, viz_outputs = None, None, None

        with torch.no_grad():
            for batch_idx, (images, masks) in enumerate(val_loader):
                images, masks = images.to(device), masks.to(device)
                outputs = model(images)

                loss = criterion(outputs, masks)
                val_losses.append(loss.item())
                val_dices.append(dice_coefficient_3d(outputs, masks).item())

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
        
        # Background distance calculation for result interpretation (No gradient impact)
        with torch.no_grad():
            per_class_hd95 = hd95_per_class_3d(viz_outputs, viz_masks)

        print(f"Epoch {epoch}/{num_epochs}")
        print(f"  TRAIN | Loss: {avg_train_loss:.4f} | Dice: {avg_train_dice:.4f}")
        print(f"  VAL   | Loss: {avg_val_loss:.4f} | Dice: {avg_val_dice:.4f} "
              f"(WT={per_class_dice[0]:.3f}, TC={per_class_dice[1]:.3f}, ET={per_class_dice[2]:.3f})")
        print(f"  DIST  | Eval HD95 (mm): WT={per_class_hd95[0]:.1f}, TC={per_class_hd95[1]:.1f}, ET={per_class_hd95[2]:.1f}")

        if epoch % viz_every == 0 or epoch == 1:
            visualize_predictions(viz_images, viz_masks, viz_outputs, epoch, loss_name, model_name)

        if avg_val_dice > best_val_dice:
            best_val_dice = avg_val_dice
            best_epoch = epoch

            save_path = f"results/models/best_{loss_name}_{model_name}_3d.pth"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_dice': best_val_dice,
                'per_class_dice': per_class_dice,
                'per_class_hd95': per_class_hd95,  # Saved alongside metrics for downstream plotting
            }, save_path)
            print(f"  [Best] New best model! Val Dice: {best_val_dice:.4f} (saved)")

        print("-" * 60)

    print(f"\nTraining complete. Best Val Dice: {best_val_dice:.4f} at Epoch {best_epoch}")
    print(f"Best model saved at: results/models/best_{loss_name}_{model_name}_3d.pth")


if __name__ == "__main__":
    print(f"\n{'='*60}")
    print(" STARTING REFINED EXPERIMENT: ATTENTION UNET + HYBRID LOSS ")
    print(f"{'='*60}")
    train_model(loss_name="hybrid", model_name="attention_unet", viz_every=5)