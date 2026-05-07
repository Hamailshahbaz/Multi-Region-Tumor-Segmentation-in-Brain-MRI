import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from data_loader import get_dataloader
from model import AttentionUNet25D

def dice_coefficient_2d(prediction, target, smooth=1e-6):
    """
    We use Dice as our 'Accuracy' metric because standard pixel accuracy 
    is misleading for imbalanced medical data.
    """
    prediction = (prediction > 0.5).float()
    intersection = (prediction * target).sum(dim=(2, 3))
    union = prediction.sum(dim=(2, 3)) + target.sum(dim=(2, 3))
    dice = (2. * intersection + smooth) / (union + smooth)
    return dice.mean()

def train_model_ce_baseline():
    # 1. Configuration
    data_dir = '/kaggle/input/datasets/awsaf49/brats20-dataset-training-validation/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_epochs = 25 
    batch_size = 16 
    learning_rate = 1e-4

    # 2. DataLoaders (Ensure these return 2.5D stacks)
    train_loader, val_loader = get_dataloader(data_dir, batch_size=batch_size)

    # 3. Model & Loss (Strictly using BCE for your proof)
    model = AttentionUNet25D(in_channels=12, out_channels=3).to(device)
    criterion = nn.BCELoss() # Binary Cross Entropy Loss
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    print(f"Running CE Baseline Experiment on {device}...")

    for epoch in range(num_epochs):
        model.train()
        train_losses, train_dices = [], []

        for i, (images, masks) in enumerate(train_loader):
            images, masks = images.to(device), masks.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            
            # Loss calculation (independent voxel log-likelihood)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()

            train_losses.append(loss.item())
            train_dices.append(dice_coefficient_2d(outputs, masks).item())

        # 4. Validation Loop
        model.eval()
        val_losses, val_dices = [], []
        
        with torch.no_grad():
            for images, masks in val_loader:
                images, masks = images.to(device), masks.to(device)
                outputs = model(images)
                
                loss = criterion(outputs, masks)
                val_losses.append(loss.item())
                val_dices.append(dice_coefficient_2d(outputs, masks).item())

        # 5. Scientific Logging
        # Note: 'Accuracy' here refers to the Dice Score (Overlap Accuracy)
        print(f"--- Epoch {epoch+1} Results ---")
        print(f"TRAIN | Loss: {np.mean(train_losses):.4f} | Accuracy (Dice): {np.mean(train_dices):.4f}")
        print(f"VAL   | Loss: {np.mean(val_losses):.4f} | Accuracy (Dice): {np.mean(val_dices):.4f}")
        print("-" * 30)

        # Always save the latest for your analysis
        torch.save(model.state_dict(), "ce_baseline_25d.pth")

if __name__ == "__main__":
    train_model_ce_baseline()