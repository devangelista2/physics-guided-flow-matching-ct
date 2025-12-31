import argparse
import os

import cv2
import numpy as np
import pandas as pd
import torch

# Metrics
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

from models.diffusion.solvers import DiffusionSolver

# Diffusion Imports
from models.diffusion.unet import load_diffusion_model

# Project Imports
from models.operators import CTOperator
from utils.tools import load_config, seed_everything


def run_experiment():
    parser = argparse.ArgumentParser(description="Run Diffusion Comparison (DPS/DDRM)")

    # Config & Paths
    parser.add_argument(
        "--config",
        type=str,
        default="configs/diffusion.yaml",
        help="Path to diffusion config",
    )
    parser.add_argument(
        "--weights", type=str, required=True, help="Path to MONAI .pth weights"
    )
    parser.add_argument(
        "--img_path", type=str, required=True, help="Path to ground truth .png"
    )
    parser.add_argument(
        "--output_dir", type=str, default="outputs/mayo_monai_finetuning"
    )

    # Physics / Data
    parser.add_argument("--img_size", type=int, default=256)
    parser.add_argument("--angles", type=int, default=180)
    parser.add_argument("--noise_sigma", type=float, default=0.01)

    # Diffusion Method Hyperparams
    parser.add_argument(
        "--dps_steps", type=int, default=50, help="Number of inference steps (e.g. 50)"
    )
    parser.add_argument("--dps_zeta", type=float, default=1.0, help="DPS step size")
    parser.add_argument(
        "--ddrm_steps", type=int, default=50, help="Number of inference steps (e.g. 50)"
    )
    parser.add_argument(
        "--cgls_steps", type=int, default=5, help="CGLS steps for DDRM correction"
    )

    # Seed
    parser.add_argument("--seed", type=int, default=42, help="Global random seed")

    args = parser.parse_args()

    # Load Config (for model params)
    config = load_config(args.config)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Set Seed
    seed_everything(args.seed)

    # --- Setup Experiment Folder ---
    exp_name = f"diff_comparison_a{args.angles}_dpsst{args.dps_steps}_zeta{args.dps_zeta}_ddrmst{args.ddrm_steps}_cgls{args.cgls_steps}"
    res_dir = os.path.join(args.output_dir, exp_name)
    os.makedirs(res_dir, exist_ok=True)

    # --- 1. Load Model ---
    print("[INFO] Loading Diffusion Model...")
    model = load_diffusion_model(args.weights, config, device=device)
    solver = DiffusionSolver(model, device=device)

    # --- 2. Load and Prepare Data ---
    print(f"[INFO] Loading image: {args.img_path}")
    # Read grayscale
    img_gt_np = cv2.imread(args.img_path, cv2.IMREAD_GRAYSCALE)
    if img_gt_np is None:
        raise ValueError("Image not found")

    img_gt_np = cv2.resize(img_gt_np, (args.img_size, args.img_size))

    # Normalize to [0, 1] for Physics
    img_gt = torch.from_numpy(img_gt_np).float() / 255.0
    img_gt = img_gt.unsqueeze(0).unsqueeze(0).to(device)  # (1, 1, H, W)

    # --- 3. Setup CT Operator ---
    print(f"[INFO] Initializing CT Operator ({args.angles} angles)...")
    # Using specific detector size if needed, else calculate dynamically
    det_size = int(args.img_size * 1.5)
    operator = CTOperator(
        img_size=args.img_size,
        num_angles=args.angles,
        detector_size=det_size,
        device=device,
    )

    # --- 4. Simulate Measurement ---
    print("[INFO] Simulating Sinogram...")
    sinogram_clean = operator.K(img_gt)

    # Add Noise (Gaussian)
    noise_level = 0.01  # 1% noise
    noise = torch.randn_like(sinogram_clean) * noise_level * sinogram_clean.max()
    sinogram_noisy = sinogram_clean + noise

    # Save Inputs (GT & Sinogram)
    cv2.imwrite(
        os.path.join(res_dir, "gt.png"),
        (img_gt.cpu().numpy()[0, 0] * 255).astype(np.uint8),
    )

    sino_vis = sinogram_noisy.cpu().numpy()[0, 0]
    sino_vis = (sino_vis - sino_vis.min()) / (sino_vis.max() - sino_vis.min()) * 255
    cv2.imwrite(os.path.join(res_dir, "sinogram_noisy.png"), sino_vis.astype(np.uint8))

    # --- Initialize Metrics ---
    psnr_fn = PeakSignalNoiseRatio(data_range=1.0).to(device)
    ssim_fn = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
    lpips_fn = LearnedPerceptualImagePatchSimilarity(net_type="vgg").to(device)

    metrics_data = []

    def compute_and_log(name, pred, target):
        """Helper to compute metrics and append to list"""
        # PSNR/SSIM expect [0, 1]
        p_val = psnr_fn(pred, target).item()
        s_val = ssim_fn(pred, target).item()

        # LPIPS expects [-1, 1] and 3 channels
        pred_3c = (pred * 2 - 1).repeat(1, 3, 1, 1).clamp(-1, 1)
        gt_3c = (target * 2 - 1).repeat(1, 3, 1, 1)
        l_val = lpips_fn(pred_3c, gt_3c).item()

        print(f"  > {name} | PSNR: {p_val:.2f}, SSIM: {s_val:.4f}, LPIPS: {l_val:.4f}")
        return {"Method": name, "PSNR": p_val, "SSIM": s_val, "LPIPS": l_val}

    # ==========================================
    # 5. Run DPS
    # ==========================================
    print(f"\n[INFO] Running DPS (Steps={args.dps_steps}, Zeta={args.dps_zeta})...")
    recon_dps = solver.sample_dps(
        measurement=sinogram_noisy,
        operator=operator,
        image_size=args.img_size,
        zeta=args.dps_zeta,
        num_inference_steps=args.dps_steps,
        clean_ref=img_gt,
        x_init=None,
        noise_scale=0.005,
    )

    # Save DPS Image
    dps_np = recon_dps.detach().cpu().numpy()[0, 0]
    cv2.imwrite(os.path.join(res_dir, "recon_dps.png"), (dps_np * 255).astype(np.uint8))

    # Compute Metrics
    metrics_data.append(compute_and_log("DPS", recon_dps, img_gt))

    # ==========================================
    # 6. Run DDRM (with CGLS)
    # ==========================================
    seed_everything(args.seed)
    print(f"\n[INFO] Running DDRM (Steps={args.ddrm_steps}, CGLS={args.cgls_steps})...")
    recon_ddrm = solver.sample_ddrm(
        measurement=sinogram_noisy,
        operator=operator,
        image_size=args.img_size,
        scale_factor=1.0,
        num_inference_steps=args.ddrm_steps,
        cgls_steps=args.cgls_steps,
        clean_ref=img_gt,
        x_init=None,
    )

    # Save DDRM Image
    ddrm_np = recon_ddrm.detach().cpu().numpy()[0, 0]
    cv2.imwrite(
        os.path.join(res_dir, "recon_ddrm.png"), (ddrm_np * 255).astype(np.uint8)
    )

    # Compute Metrics
    metrics_data.append(compute_and_log("DDRM", recon_ddrm, img_gt))

    # --- 7. Save Metrics CSV ---
    df = pd.DataFrame(metrics_data)
    csv_path = os.path.join(res_dir, "metrics.csv")
    df.to_csv(csv_path, index=False)

    print("-" * 30)
    print(f"Comparison Complete. Results saved to: {res_dir}")

    # Cleanup ASTRA
    operator.cleanup()


if __name__ == "__main__":
    run_experiment()
