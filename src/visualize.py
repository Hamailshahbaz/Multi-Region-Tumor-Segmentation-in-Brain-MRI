import os
import sys
import torch
import numpy as np
import nibabel as nib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from data_loader import BraTSDataset3D


def visualize_predictions(images, masks, outputs, epoch, loss_name, model_name, out_dir="results/plots"):
    os.makedirs(out_dir, exist_ok=True)
    img = images[0].cpu().numpy()                     # (4, H, W, D)
    mask = masks[0].cpu().numpy()                     # (3, H, W, D)
    pred = outputs[0].cpu().detach().numpy()          # (3, H, W, D)
    pred_bin = (pred > 0.5).astype(np.float32)

    # Extract middle axial slice for plotting
    mid = img.shape[-1] // 2
    fig, axes = plt.subplots(3, 3, figsize=(10, 10))
    class_names = ["Whole Tumor (WT)", "Tumor Core (TC)", "Enhancing Tumor (ET)"]

    for i in range(3):
        axes[i, 0].imshow(img[1, :, :, mid], cmap='gray')  # FLAIR middle slice
        axes[i, 0].set_title(f"Input (FLAIR) - {class_names[i]}")
        axes[i, 0].axis('off')

        axes[i, 1].imshow(mask[i, :, :, mid], cmap='jet', vmin=0, vmax=1)
        axes[i, 1].set_title("Ground Truth")
        axes[i, 1].axis('off')

        axes[i, 2].imshow(pred_bin[i, :, :, mid], cmap='jet', vmin=0, vmax=1)
        axes[i, 2].set_title(f"Prediction (Dice-based)")
        axes[i, 2].axis('off')

    plt.tight_layout()
    save_path = os.path.join(out_dir, f"{loss_name}_{model_name}_epoch{epoch:03d}.png")
    plt.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"  [Vis] Saved prediction plot: {save_path}")


#  STANDALONE EVALUATION VISUALIZER
#  Runs when:  python src/visualize.py
#  Evaluates EVAL_PERCENT% of the dataset and saves one
#  3×4 grid PNG per patient to results/evaluation_plots/

# ── Configuration ─────────────────────────────────────────────
_WEIGHTS_PATH   = "results/models/AttentionUnet/new/best_hybrid_attention_unet_3d.pth"
_DATA_DIR       = os.path.expanduser(
    "~/Multi-Region-Tumor-Segmentation-in-Brain-MRI/data/"
    "BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData"
)
_OUTPUT_DIR     = "results/evaluation_plots"
_EVAL_PERCENT   = 10        # % of patients to evaluate (e.g. 10 = 10%)
_PROB_THRESHOLD = 0.5
_CROP_SIZE      = (128, 128, 128)
_RANDOM_SEED    = 42


# ── Helpers ───────────────────────────────────────────────────

def _load_nifti(patient_dir, patient_id, modality):
    path = os.path.join(patient_dir, f"{patient_id}_{modality}.nii")
    if not os.path.exists(path):
        path += ".gz"
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing '{modality}' in {patient_dir}")
    return nib.load(path).get_fdata().astype(np.float32)


def _center_crop(vol, target=_CROP_SIZE):
    h, w, d = vol.shape
    th, tw, td = target
    sh, sw, sd = (h - th) // 2, (w - tw) // 2, (d - td) // 2
    return vol[sh:sh+th, sw:sw+tw, sd:sd+td]


def _zscore(vol):
    s = vol.std()
    return (vol - vol.mean()) / s if s > 0 else vol


def _auto_threshold(probs, thresh):
    """Binary mask; falls back to top-5% percentile if max prob < thresh."""
    preds = np.zeros_like(probs, dtype=np.float32)
    for c in range(probs.shape[0]):
        ch_max = probs[c].max()
        if ch_max >= thresh:
            preds[c] = (probs[c] >= thresh).astype(np.float32)
        elif ch_max > 0:
            t = np.percentile(probs[c], 95)
            preds[c] = (probs[c] >= t).astype(np.float32)
            print(f"    [WARN] Ch{c}: max={ch_max:.3f} < {thresh}, using p95={t:.3f}")
    return preds


def _best_axial_slice(prob_wt, gt_wt):
    sums = gt_wt.sum(axis=(0, 1)) if gt_wt.sum() > 0 else prob_wt.sum(axis=(0, 1))
    return int(np.argmax(sums)) if sums.max() > 0 else prob_wt.shape[2] // 2


def _overlay_mask(ax, backdrop, mask, cmap_name, label, alpha=0.55):
    ax.imshow(backdrop, cmap='gray', interpolation='bilinear')
    if mask.sum() > 0:
        cmap_obj = plt.get_cmap(cmap_name)
        colored  = cmap_obj(mask)
        colored[..., 3] = mask * alpha
        ax.imshow(colored, interpolation='bilinear')
        status, color = "DETECTED",     "#00e676"
    else:
        status, color = "NOT DETECTED", "#ff5252"
    ax.set_title(label, fontsize=9, color='white', pad=3)
    ax.text(0.5, 0.03, status, transform=ax.transAxes,
            ha='center', va='bottom', fontsize=8, fontweight='bold', color=color,
            bbox=dict(boxstyle='round,pad=0.2', facecolor='#111111', alpha=0.7))
    ax.axis('off')


def _dice2d(p, g, eps=1e-6):
    return (2 * (p * g).sum() + eps) / (p.sum() + g.sum() + eps)


def run_evaluation():
    from model import AttentionUNet3D

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    print("Loading model ...")
    model = AttentionUNet3D(in_channels=4, out_channels=3).to(device)
    ckpt  = torch.load(_WEIGHTS_PATH, map_location=device, weights_only=False)
    model.load_state_dict(ckpt.get("model_state_dict", ckpt))
    model.eval()
    print("Model loaded\n")

    # Discover patients
    if not os.path.isdir(_DATA_DIR):
        sys.exit(f"[ERROR] DATA_DIR not found:\n  {_DATA_DIR}\n"
                 f"Update _DATA_DIR in visualize.py")

    all_patients = sorted([
        p for p in os.listdir(_DATA_DIR)
        if os.path.isdir(os.path.join(_DATA_DIR, p)) and p.startswith("BraTS20")
    ])
    if not all_patients:
        sys.exit(f"[ERROR] No BraTS20 folders found in:\n  {_DATA_DIR}")

    n_total  = len(all_patients)
    n_sample = max(1, int(np.ceil(n_total * _EVAL_PERCENT / 100)))
    rng      = np.random.default_rng(_RANDOM_SEED)
    selected = sorted(rng.choice(all_patients, size=n_sample, replace=False).tolist())

    print(f"Dataset : {n_total} patients total")
    print(f"Sampling: {_EVAL_PERCENT}%  ->  {n_sample} patients")
    print(f"Output  : {_OUTPUT_DIR}/\n")
    print("-" * 55)

    os.makedirs(_OUTPUT_DIR, exist_ok=True)
    summary = {"WT": [], "TC": [], "ET": [], "skipped": []}

    FIG_BG    = '#0d0d0d'
    ROW_NAMES = ["Whole Tumor (WT)", "Tumor Core (TC)", "Enhancing Tumor (ET)"]
    GT_CMAPS  = ['Reds',   'Oranges', 'Purples']
    PR_CMAPS  = ['YlOrRd', 'YlOrBr',  'RdPu']
    HM_CMAPS  = ['hot',    'copper',  'magma']

    for idx, patient_id in enumerate(selected, 1):
        patient_dir = os.path.join(_DATA_DIR, patient_id)
        print(f"[{idx:>3}/{n_sample}]  {patient_id}", end="  ...  ", flush=True)

        # 1. Load & preprocess
        try:
            vols = []
            for mod in ["flair", "t1ce", "t1", "t2"]:
                v = _load_nifti(patient_dir, patient_id, mod)
                v = _center_crop(v)
                v = _zscore(v)
                vols.append(v)
            image_np = np.stack(vols, axis=0)           # (4, H, W, D)

            t1ce_raw = _center_crop(
                _load_nifti(patient_dir, patient_id, "t1ce")
            )
            seg_raw  = _center_crop(
                _load_nifti(patient_dir, patient_id, "seg")
            )
            gt_wt = (seg_raw > 0).astype(np.float32)
            gt_tc = np.logical_or(seg_raw == 1, seg_raw == 4).astype(np.float32)
            gt_et = (seg_raw == 4).astype(np.float32)

        except FileNotFoundError as e:
            print(f"SKIPPED  ({e})")
            summary["skipped"].append(patient_id)
            continue

        # 2. Inference
        tensor_in = torch.from_numpy(image_np).float().unsqueeze(0).to(device)
        with torch.no_grad():
            probs = model(tensor_in).squeeze(0).cpu().numpy()   # (3, H, W, D)

        preds = _auto_threshold(probs, _PROB_THRESHOLD)

        # 3. Best slice & 2-D extraction
        sl = _best_axial_slice(probs[0], gt_wt)

        def s2d(vol):
            return np.rot90(vol[:, :, sl])

        backdrop   = s2d(t1ce_raw)
        gt_masks   = [s2d(gt_wt),    s2d(gt_tc),    s2d(gt_et)]
        pred_masks = [s2d(preds[0]), s2d(preds[1]), s2d(preds[2])]
        prob_maps  = [s2d(probs[0]), s2d(probs[1]), s2d(probs[2])]

        d_wt = _dice2d(pred_masks[0], gt_masks[0])
        d_tc = _dice2d(pred_masks[1], gt_masks[1])
        d_et = _dice2d(pred_masks[2], gt_masks[2])
        summary["WT"].append(d_wt)
        summary["TC"].append(d_tc)
        summary["ET"].append(d_et)

        # 4. Plot 3×4 grid
        dice_vals = [d_wt, d_tc, d_et]
        fig, axes = plt.subplots(3, 4, figsize=(20, 15))
        fig.patch.set_facecolor(FIG_BG)
        fig.suptitle(
            f"Patient: {patient_id}  |  Axial Slice: {sl}  |  "
            f"Dice  WT={d_wt:.3f}  TC={d_tc:.3f}  ET={d_et:.3f}",
            color='white', fontsize=13, fontweight='bold', y=0.998
        )

        col_titles = ["Base T1ce MRI", "GT Mask Overlay",
                      "Pred Mask Overlay", "Pred Probability Map"]
        for ci, ct in enumerate(col_titles):
            axes[0, ci].set_title(ct, color='#aaaaaa', fontsize=10, pad=5)

        for row in range(3):
            for ax in axes[row]:
                ax.set_facecolor(FIG_BG)

            # Col 0: base MRI
            axes[row, 0].imshow(backdrop, cmap='gray')
            axes[row, 0].set_ylabel(ROW_NAMES[row], color='white',
                                    fontsize=10, labelpad=5)
            axes[row, 0].set_xticks([]); axes[row, 0].set_yticks([])
            for sp in axes[row, 0].spines.values():
                sp.set_edgecolor('#444444')

            # Col 1: GT overlay
            _overlay_mask(axes[row, 1], backdrop, gt_masks[row],
                          GT_CMAPS[row], label=f"GT: {ROW_NAMES[row]}")

            # Col 2: Pred overlay
            _overlay_mask(axes[row, 2], backdrop, pred_masks[row],
                          PR_CMAPS[row],
                          label=f"Pred: {ROW_NAMES[row]}\n2D Dice={dice_vals[row]:.3f}")

            # Col 3: Probability heatmap
            axes[row, 3].imshow(backdrop, cmap='gray', alpha=0.4)
            im = axes[row, 3].imshow(prob_maps[row], cmap=HM_CMAPS[row],
                                      vmin=0, vmax=1, alpha=0.85,
                                      interpolation='bilinear')
            cbar = fig.colorbar(im, ax=axes[row, 3], fraction=0.046, pad=0.04)
            cbar.ax.yaxis.set_tick_params(color='white')
            plt.setp(cbar.ax.yaxis.get_ticklabels(), color='white', fontsize=7)
            axes[row, 3].set_title(
                f"Prob Map: {ROW_NAMES[row]}\nmax={prob_maps[row].max():.3f}",
                color='white', fontsize=9, pad=3)
            axes[row, 3].axis('off')

        plt.tight_layout(rect=[0, 0, 1, 0.997])
        out_path = os.path.join(_OUTPUT_DIR, f"{patient_id}_eval.png")
        plt.savefig(out_path, dpi=150, bbox_inches='tight',
                    facecolor=FIG_BG, edgecolor='none')
        plt.close(fig)
        print(f"WT={d_wt:.3f}  TC={d_tc:.3f}  ET={d_et:.3f}  ->  saved")

    # Summary
    print("\n" + "=" * 55)
    print("EVALUATION SUMMARY")
    print("=" * 55)
    done = len(summary["WT"])
    if done:
        print(f"Patients evaluated : {done}")
        print(f"Mean Dice  WT : {np.mean(summary['WT']):.4f}  (+-{np.std(summary['WT']):.4f})")
        print(f"Mean Dice  TC : {np.mean(summary['TC']):.4f}  (+-{np.std(summary['TC']):.4f})")
        print(f"Mean Dice  ET : {np.mean(summary['ET']):.4f}  (+-{np.std(summary['ET']):.4f})")
    else:
        print("No patients were successfully evaluated.")
    if summary["skipped"]:
        print(f"\nSkipped ({len(summary['skipped'])}):")
        for p in summary["skipped"]:
            print(f"  {p}")
    print(f"\nPlots saved to: {_OUTPUT_DIR}/")


if __name__ == "__main__":
    run_evaluation()
