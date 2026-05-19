import os
import glob
import torch
import numpy as np
import nibabel as nib
import gradio as gr
import matplotlib.pyplot as plt
from model import AttentionUNet3D

# --- CONFIGURATION ---
# Replace with your actual model weights path
WEIGHTS_PATH = "/results/models/AttentionUnet/best_hybrid_attention_unet_3d.pth"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 1. Initialize and Load Model globally
print("🧠 Loading model into memory...")
model = AttentionUNet3D(in_channels=4, out_channels=3).to(device)
checkpoint = torch.load(WEIGHTS_PATH, map_location=device, weights_only=False)
if "model_state_dict" in checkpoint:
    model.load_state_dict(checkpoint["model_state_dict"])
else:
    model.load_state_dict(checkpoint)
model.eval()
print("✅ Model weights loaded successfully!")

def segment_all_channels(nifti_file):
    """
    Inference function modified to display all 3 channels in a 4-panel grid.
    """
    if nifti_file is None:
        return None, "Please upload a valid NIfTI file."
    
    try:
        # Load the uploaded 3D volume
        img = nib.load(nifti_file.name)
        data = img.get_fdata()
        
        # BRAINS simulation for demo: use input data for all 4 required channels
        if len(data.shape) == 3:
            x_mat = np.stack([data, data, data, data], axis=0)
        elif len(data.shape) == 4 and data.shape[0] == 4:
            x_mat = data
        else:
            return None, "Unsupported shape format. Expected a 3D or 4D stacked NIfTI volume."
            
        # Z-score normalization per channel
        for i in range(x_mat.shape[0]):
            if np.std(x_mat[i]) > 0:
                x_mat[i] = (x_mat[i] - np.mean(x_mat[i])) / np.std(x_mat[i])
                
        # Convert to tensor & add batch dim -> (1, 4, H, W, D)
        tensor_in = torch.tensor(x_mat).float().unsqueeze(0).to(device)
        
        # Run inference (raw logits)
        with torch.no_grad():
            probs = model(tensor_in).squeeze(0) # Shape: (3, H, W, D) directly has probabilities
            preds = (probs > 0.5).float()
            
        # Convert to numpy for visualization
        probs_np = probs.cpu().numpy()
        preds_np = preds.cpu().numpy()
        raw_mri = data # original volume backdrop
        
        # Find best axial slice (largest WT density)
        wt_sum_per_slice = np.sum(preds_np[0], axis=(0, 1))
        mid_slice_idx = np.argmax(wt_sum_per_slice) if np.max(wt_sum_per_slice) > 0 else raw_mri.shape[2] // 2
        # Extract 2D slices
        backdrop_slice = np.rot90(raw_mri[:, :, mid_slice_idx])
        wt_mask = np.rot90(preds_np[0, :, :, mid_slice_idx])
        tc_mask = np.rot90(preds_np[1, :, :, mid_slice_idx])
        et_mask = np.rot90(preds_np[2, :, :, mid_slice_idx])
        
        # --- Create 4-PANEL Grid Visualization ---
        # Mirrors input layout: [Base MRI, WT, TC, ET]
        fig, axes = plt.subplots(1, 4, figsize=(20, 5))
        plt.subplots_adjust(wspace=0.1) # tighten spacing
        
        # Configuration for colors and names
        region_names = ["Whole Tumor (WT)", "Tumor Core (TC)", "Enhancing Tumor (ET)"]
        region_colors = ['Reds', 'Oranges', 'Purples']
        
        # 1. Plot Base MRI
        axes[0].imshow(backdrop_slice, cmap='gray')
        axes[0].set_title(f"Base MRI\n(Slice {mid_slice_idx})", fontsize=14)
        axes[0].axis('off')
        
        # 2-4. Plot Segments with MRI backdrop
        masks = [wt_mask, tc_mask, et_mask]
        for i in range(3):
            ax = axes[i+1]
            mask = masks[i]
            
            # Show gray MRI backdrop first
            ax.imshow(backdrop_slice, cmap='gray')
            
            # Layer the prediction with transparency if tumor detected
            if np.sum(mask) > 0:
                ax.imshow(mask, cmap=region_colors[i], alpha=0.5)
                det_status = "DETECTED"
            else:
                det_status = "NOT DETECTED"
                
            ax.set_title(f"Pred: {region_names[i]}\n({det_status})", fontsize=14, color='white' if det_status == "DETECTED" else 'gray')
            ax.axis('off')
            
        plt.tight_layout()
        
        # Save tmp plot
        plot_path = "tmp_inference_grid.png"
        plt.savefig(plot_path, bbox_inches='tight', pad_inches=0.1, dpi=150, facecolor='#111111') # darken background
        plt.close()
        
        # --- CALC REGIONAL CONFIDENCE (Keep analytical robustness) ---
        def get_region_confidence(prob_channel, pred_mask):
            masked_probs = prob_channel[pred_mask == 1]
            if len(masked_probs) > 0:
                return float(np.mean(masked_probs))
            return 0.0

        conf_wt = get_region_confidence(probs_np[0], preds_np[0])
        conf_tc = get_region_confidence(probs_np[1], preds_np[1])
        conf_et = get_region_confidence(probs_np[2], preds_np[2])
        
        metrics_output = (
            f"🎯 TARGET AXIAL SLICE SHOWN: {mid_slice_idx}\n"
            f"----------------------------------------\n"
            f"🔴 WT Presence: {conf_wt * 100:.2f}% Confidence\n\n"
            f"🟠 TC Presence: {conf_tc * 100:.2f}% Confidence\n\n"
            f"🟣 ET Presence: {conf_et * 100:.2f}% Confidence\n"
        )
        
        return plot_path, metrics_output

    except Exception as e:
        return None, f"❌ Pipeline execution error: {str(e)}"

# --- BUILD UPDATED GRADIO INTERFACE ---
# Use dark theme to match clinical aesthetic
with gr.Blocks(theme=gr.themes.Default(primary_hue="teal", neutral_hue="slate")) as demo:
    gr.Markdown(
        """
        # 🧠 3D Attention U-Net: Four-Panel Clinician Visualizer
        ### Volumetric Brain MRI Segmentation Engine. View individual predictions for WT, TC, and ET alongside structural context.
        """
    )
    
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### 📂 Input Selection")
            file_input = gr.File(label="Upload Patient NIfTI Volume (.nii / .nii.gz)", file_types=[".nii", ".nii.gz"])
            submit_btn = gr.Button("🚀 Run AI Inference", variant="primary")
            
            # Analytical Textbox relocated to input column for cleaner grid display
            text_output = gr.Textbox(label="Confidence Scoring & Status", lines=10, interactive=False)
            
        with gr.Column(scale=4): # scale up visual column
            gr.Markdown("### 📊 Segmentation Visualizer Grid (Base MRI + Individual Channels)")
            # The Image component now accepts a 4-panel grid graphic
            image_output = gr.Image(label="Visualization Grid", type="filepath", interactive=False)

    # Wire button
    submit_btn.click(
        fn=segment_all_channels,
        inputs=file_input,
        outputs=[image_output, text_output]
    )

if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7860, share=False)