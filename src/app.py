import os
import numpy as np
import nibabel as nib
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import gradio as gr
from model import AttentionUNet3D

#  CONFIGURATION
WEIGHTS_PATH   = "results/models/AttentionUnet/best_hybrid_attention_unet_3d.pth"
PROB_THRESHOLD = 0.5
CROP_SIZE      = (128, 128, 128)
TMP_PLOT_PATH  = "tmp_inference_grid.png"
FIG_BG         = '#0d0d0d'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#  LOAD MODEL  (once at startup)
print("Loading model ...")
model = AttentionUNet3D(in_channels=4, out_channels=3).to(device)
ckpt  = torch.load(WEIGHTS_PATH, map_location=device, weights_only=False)
model.load_state_dict(ckpt.get("model_state_dict", ckpt))
model.eval()
print("Model loaded successfully!")


#  HELPERS

def center_crop(vol, target=CROP_SIZE):
    h, w, d = vol.shape
    th, tw, td = target
    sh, sw, sd = (h - th) // 2, (w - tw) // 2, (d - td) // 2
    return vol[sh:sh+th, sw:sw+tw, sd:sd+td]


def zscore(vol):
    s = vol.std()
    return (vol - vol.mean()) / s if s > 0 else vol


def auto_threshold(probs, thresh=PROB_THRESHOLD):
    """
    Binary mask with top-5% percentile fallback when model never reaches thresh.
    This ensures we always show something if the model has any activations.
    """
    preds = np.zeros_like(probs, dtype=np.float32)
    warn  = []
    for c in range(probs.shape[0]):
        ch_max = probs[c].max()
        if ch_max >= thresh:
            preds[c] = (probs[c] >= thresh).astype(np.float32)
        elif ch_max > 0:
            t = np.percentile(probs[c], 95)
            preds[c] = (probs[c] >= t).astype(np.float32)
            warn.append(
                f"Ch{c}: max_prob={ch_max:.3f} < {thresh} → fallback p95={t:.3f}"
            )
    return preds, warn


def best_axial_slice(prob_wt, preds_wt):
    """Use predicted WT mask first, fall back to raw probability sums."""
    sums = preds_wt.sum(axis=(0, 1))
    if sums.max() == 0:
        sums = prob_wt.sum(axis=(0, 1))
    return int(np.argmax(sums)) if sums.max() > 0 else prob_wt.shape[2] // 2


def overlay_mask(ax, backdrop, mask, cmap_name, label, alpha=0.55):
    ax.imshow(backdrop, cmap='gray', interpolation='bilinear')
    if mask.sum() > 0:
        cmap_obj = plt.get_cmap(cmap_name)
        colored  = cmap_obj(mask)
        colored[..., 3] = mask * alpha
        ax.imshow(colored, interpolation='bilinear')
        status, color = "DETECTED",     "#00e676"
    else:
        status, color = "NOT DETECTED", "#ff5252"
    ax.set_title(label, fontsize=5, color='white', pad=4)
    ax.text(0.5, 0.03, status, transform=ax.transAxes,
            ha='center', va='bottom', fontsize=9, fontweight='bold', color=color,
            bbox=dict(boxstyle='round,pad=0.2', facecolor='#111111', alpha=0.7))
    ax.axis('off')


def get_confidence(prob_ch, pred_mask):
    masked = prob_ch[pred_mask == 1]
    return float(np.mean(masked)) if len(masked) > 0 else 0.0


#  INFERENCE FUNCTION
def segment(flair_file, t1_file, t1ce_file, t2_file):
    """
    Accepts the 4 BraTS modality NIfTI files separately.
    Returns: (plot_path, metrics_text)
    """
    # ── Validate inputs ──────────────────────────────────────
    files = {"FLAIR": flair_file, "T1": t1_file, "T1ce": t1ce_file, "T2": t2_file}
    missing = [k for k, v in files.items() if v is None]
    if missing:
        return None, f"Please upload all 4 modalities. Missing: {', '.join(missing)}"

    try:
        # ── 1. Load & preprocess each modality ───────────────
        vols = []
        for label, f in files.items():
            data = nib.load(f.name).get_fdata().astype(np.float32)
            if data.ndim != 3:
                return None, f"{label}: expected a 3D volume, got shape {data.shape}"
            data = center_crop(data)
            data = zscore(data)
            vols.append(data)

        image_np = np.stack(vols, axis=0)           # (4, H, W, D)

        # T1ce as backdrop (un-normalised for better contrast)
        t1ce_raw = center_crop(
            nib.load(t1ce_file.name).get_fdata().astype(np.float32)
        )

        # ── 2. Inference ──────────────────────────────────────
        tensor_in = torch.from_numpy(image_np).float().unsqueeze(0).to(device)
        with torch.no_grad():
            probs = model(tensor_in).squeeze(0).cpu().numpy()  # (3, H, W, D)

        preds, warnings = auto_threshold(probs)

        # ── 3. Best axial slice ───────────────────────────────
        sl = best_axial_slice(probs[0], preds[0])

        def s2d(vol):
            return np.rot90(vol[:, :, sl])

        backdrop   = s2d(t1ce_raw)
        gt_or_base = backdrop                           # no GT available in app
        pred_masks = [s2d(preds[i]) for i in range(3)]
        prob_maps  = [s2d(probs[i]) for i in range(3)]

        # ── 4. Confidence scores ──────────────────────────────
        conf = [get_confidence(probs[i], preds[i]) for i in range(3)]

        # ── 5. Build 3×4 figure ───────────────────────────────
        #   Col 0 : Base T1ce MRI
        #   Col 1 : Pred mask overlay  (WT / TC / ET per row)
        #   Col 2 : Probability heatmap
        #   Col 3 : Summary panel (text stats)
        ROW_NAMES = ["Whole Tumor (WT)", "Tumor Core (TC)", "Enhancing Tumor (ET)"]
        PR_CMAPS  = ['YlOrRd', 'YlOrBr', 'RdPu']
        HM_CMAPS  = ['hot',    'copper',  'magma']
        ROW_ICONS = ['🔴', '🟠', '🟣']

        fig, axes = plt.subplots(3, 3, figsize=(5, 5))
        fig.patch.set_facecolor(FIG_BG)
        fig.suptitle(
            f"3D Attention U-Net  |  Brain Tumor Segmentation  |  Axial Slice: {sl}",
            color='white', fontsize=12, fontweight='bold', y=0.999
        )

        col_titles = ["Base T1ce MRI", "Pred Mask Overlay", "Pred Probability Map"]
        for ci, ct in enumerate(col_titles):
            axes[0, ci].set_title(ct, color='#aaaaaa', fontsize=5, pad=6)

        for row in range(3):
            for ax in axes[row]:
                ax.set_facecolor(FIG_BG)

            # Col 0 — Base MRI (same for all rows, with row label)
            axes[row, 0].imshow(backdrop, cmap='gray')
            axes[row, 0].set_ylabel(ROW_NAMES[row], color='white',
                                    fontsize=5, labelpad=6)
            axes[row, 0].set_xticks([]); axes[row, 0].set_yticks([])
            for sp in axes[row, 0].spines.values():
                sp.set_edgecolor('#444444')

            # Col 1 — Prediction overlay
            overlay_mask(
                axes[row, 1], backdrop, pred_masks[row], PR_CMAPS[row],
                label=f"Pred: {ROW_NAMES[row]}\nConf: {conf[row]*100:.1f}%"
            )

            # Col 2 — Probability heatmap (always visible)
            axes[row, 2].imshow(backdrop, cmap='gray', alpha=0.35)
            im = axes[row, 2].imshow(
                prob_maps[row], cmap=HM_CMAPS[row],
                vmin=0, vmax=1, alpha=0.88, interpolation='bilinear'
            )
            cbar = fig.colorbar(im, ax=axes[row, 2], fraction=0.046, pad=0.04)
            cbar.ax.yaxis.set_tick_params(color='white')
            plt.setp(cbar.ax.yaxis.get_ticklabels(), color='white', fontsize=5)
            axes[row, 2].set_title(
                f"Prob Map: {ROW_NAMES[row]}\nmax={prob_maps[row].max():.3f}",
                color='white', fontsize=5, pad=4
            )
            axes[row, 2].axis('off')

        plt.tight_layout(rect=[0, 0, 1, 0.998])
        plt.savefig(TMP_PLOT_PATH, dpi=150, bbox_inches='tight',
                    facecolor=FIG_BG, edgecolor='none')
        plt.close(fig)

        # ── 6. Build metrics text ─────────────────────────────
        region_labels = ["WT (Whole Tumor)", "TC (Tumor Core)", "ET (Enhancing Tumor)"]
        lines = [
            f"Target axial slice shown : {sl}",
            "─" * 42,
        ]
        for i, (icon, label, c) in enumerate(zip(ROW_ICONS, region_labels, conf)):
            detected = pred_masks[i].sum() > 0
            status   = "DETECTED" if detected else "NOT DETECTED"
            lines.append(f"{icon}  {label}")
            lines.append(f"    Status     : {status}")
            lines.append(f"    Confidence : {c * 100:.2f}%")
            lines.append(f"    Max prob   : {probs[i].max():.4f}")
            lines.append("")

        if warnings:
            lines.append("⚠️  Threshold warnings:")
            lines.extend(f"   {w}" for w in warnings)

        return TMP_PLOT_PATH, "\n".join(lines)

    except Exception as e:
        import traceback
        return None, f"Pipeline error:\n{traceback.format_exc()}"


#  GRADIO UI
with gr.Blocks(
    theme=gr.themes.Default(primary_hue="teal", neutral_hue="slate"),
    title="Brain Tumor Segmentation"
) as demo:

    gr.Markdown("""
    # 🧠 3D Attention U-Net — Brain Tumor Segmentation
    ### Upload all **4 BraTS modalities** separately. The model predicts WT, TC, and ET regions.
    > **Modalities required:** FLAIR · T1 · T1ce · T2 — each as a separate `.nii` or `.nii.gz` file.
    """)

    with gr.Row():
        # ── Input column ────────────────────────────────────
        with gr.Column(scale=1):
            gr.Markdown("### 📂 Upload Modalities")
            flair_input = gr.File(label="FLAIR (.nii / .nii.gz)",  file_types=[".nii", ".nii.gz"])
            t1_input    = gr.File(label="T1    (.nii / .nii.gz)",  file_types=[".nii", ".nii.gz"])
            t1ce_input  = gr.File(label="T1ce  (.nii / .nii.gz)",  file_types=[".nii", ".nii.gz"])
            t2_input    = gr.File(label="T2    (.nii / .nii.gz)",  file_types=[".nii", ".nii.gz"])
            run_btn     = gr.Button("🚀 Run Segmentation", variant="primary")

            gr.Markdown("### 📊 Detection Report")
            text_output = gr.Textbox(
                label="Confidence Scores & Status",
                lines=18, interactive=False
            )

        # ── Output column ────────────────────────────────────
        with gr.Column(scale=3):
            gr.Markdown("### 🖼️ Segmentation Grid  (Base MRI · Pred Mask · Prob Map)")
            image_output = gr.Image(
                label="Visualization Grid",
                type="filepath", interactive=False
            )

    run_btn.click(
        fn=segment,
        inputs=[flair_input, t1_input, t1ce_input, t2_input],
        outputs=[image_output, text_output]
    )

if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7860, share=False)