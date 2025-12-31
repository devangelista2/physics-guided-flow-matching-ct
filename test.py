import argparse
import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

# Metrics
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from tqdm import tqdm

from models.operators import CTOperator
from models.solvers import ReconstructionSolver

# Custom modules
from models.unet import UNet
from utils.tools import load_config, save_grid, seed_everything


def run_experiment():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    parser.add_argument(
        "--weights", type=str, required=True, help="Path to trained .pth"
    )
    parser.add_argument(
        "--img_path", type=str, required=True, help="Path to ground truth .png"
    )
    parser.add_argument(
        "--method",
        type=str,
        default="flowdps",
        choices=["flowdps", "pnp", "ictm", "flowers"],
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=50,
        help="Number of steps of the Flow Matching Process",
    )
    parser.add_argument("--scale", type=float, default=1.0, help="Guidance scale")
    parser.add_argument("--angles", type=int, default=180, help="Number of CT angles")
    parser.add_argument("--seed", type=int, default=1, help="Global random seed")
    args = parser.parse_args()

    config = load_config(args.config)
    device = torch.device(config["device"])
    seed_everything(args.seed)

    # --- Setup Experiment Folder ---
    exp_name = f"recon_{args.method}_st{args.steps}_a{args.angles}_s{args.scale}"
    res_dir = os.path.join("outputs", config["experiment_name"], exp_name)
    snap_dir = os.path.join(res_dir, "snapshots")
    os.makedirs(snap_dir, exist_ok=True)

    # --- 1. Load Model ---
    print("[INFO] Loading Flow Matching Model...")
    model = UNet(config).to(device)
    model.load_state_dict(torch.load(args.weights, map_location=device))
    model.eval()

    # --- 2. Load and Prepare Data ---
    print(f"[INFO] Loading image: {args.img_path}")
    # Read grayscale
    img_gt_np = cv2.imread(args.img_path, cv2.IMREAD_GRAYSCALE)
    if img_gt_np is None:
        raise ValueError("Image not found")

    img_gt_np = cv2.resize(
        img_gt_np, (config["data"]["img_size"], config["data"]["img_size"])
    )

    # Normalize to [0, 1] for Physics
    img_gt = torch.from_numpy(img_gt_np).float() / 255.0
    img_gt = img_gt.unsqueeze(0).unsqueeze(0).to(device)  # (1, 1, H, W)

    # --- 3. Setup CT Operator ---
    print("[INFO] Initializing CT Operator...")
    operator = CTOperator(
        img_size=config["data"]["img_size"],
        num_angles=args.angles,
        detector_size=int(config["data"]["img_size"] * 1.5),  # Enough coverage
        device=device,
    )

    # --- 4. Simulate Measurement ---
    print("[INFO] Simulating Sinogram...")
    sinogram_clean = operator.K(img_gt)

    # Add Noise (Gaussian)
    noise_level = 0.01  # 1% noise
    noise = torch.randn_like(sinogram_clean) * noise_level * sinogram_clean.max()
    sinogram_noisy = sinogram_clean + noise

    # Save Inputs
    cv2.imwrite(
        os.path.join(res_dir, "gt.png"),
        (img_gt.cpu().numpy()[0, 0] * 255).astype(np.uint8),
    )

    sino_vis = sinogram_noisy.cpu().numpy()[0, 0]
    sino_vis = (sino_vis - sino_vis.min()) / (sino_vis.max() - sino_vis.min()) * 255
    cv2.imwrite(os.path.join(res_dir, "sinogram.png"), sino_vis.astype(np.uint8))

    # --- 5. Reconstruction ---
    print(f"[INFO] Running Reconstruction ({args.method})...")
    solver = ReconstructionSolver(model, operator, config)

    # We solve!
    # Note: solver returns image in [0, 1]
    recon_img, snapshots = solver.solve(
        sinogram_noisy,
        method=args.method,
        steps=args.steps,
        scale=args.scale,
        save_snapshots=True,
        save_path=res_dir,
        noise_scale=0.005,
    )

    # Save Result
    recon_np = recon_img.cpu().numpy()[0, 0]
    cv2.imwrite(os.path.join(res_dir, "recon.png"), (recon_np * 255).astype(np.uint8))

    # --- 6. Metrics Calculation ---
    print("[INFO] Computing Metrics...")

    # Initialize Metrics
    psnr_fn = PeakSignalNoiseRatio(data_range=1.0).to(device)
    ssim_fn = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
    lpips_fn = LearnedPerceptualImagePatchSimilarity(net_type="vgg").to(device)

    metrics_data = []

    # We compute metrics for every snapshot to trace convergence
    # GT needs to be repeated to match snapshot shape if we processed batch,
    # but here batch is 1.

    # Note: LPIPS expects inputs in [-1, 1], others in [0, 1] usually or match data_range
    # My snapshots are in [-1, 1] (raw model output), need conversion

    for i, snap in enumerate(snapshots):
        # Snap is (B, 1, H, W) in range [-1, 1] (approx)
        snap = snap.to(device)
        snap_01 = (snap.clamp(-1, 1) + 1) / 2

        # Calculate
        p_val = psnr_fn(snap_01, img_gt).item()
        s_val = ssim_fn(snap_01, img_gt).item()

        # LPIPS needs 3 channels and [-1, 1] range
        snap_3c = snap.repeat(1, 3, 1, 1).clamp(-1, 1)
        gt_3c = (img_gt * 2 - 1).repeat(1, 3, 1, 1)  # GT to [-1, 1]
        l_val = lpips_fn(snap_3c, gt_3c).item()

        metrics_data.append(
            {"iteration": i * (50 // 10), "PSNR": p_val, "SSIM": s_val, "LPIPS": l_val}
        )

        # Save Snapshot image
        save_grid(snap_01, os.path.join(snap_dir, f"iter_{i}.png"))

    # Final Metric
    final_psnr = psnr_fn(recon_img, img_gt).item()
    final_ssim = ssim_fn(recon_img, img_gt).item()

    recon_3c = (recon_img * 2 - 1).repeat(1, 3, 1, 1)
    final_lpips = lpips_fn(recon_3c, gt_3c).item()

    metrics_data.append(
        {"iteration": 50, "PSNR": final_psnr, "SSIM": final_ssim, "LPIPS": final_lpips}
    )

    # Save CSV
    df = pd.DataFrame(metrics_data)
    df.to_csv(os.path.join(res_dir, "metrics.csv"), index=False)

    print("-" * 30)
    print(f"Final Results:")
    print(f"PSNR: {final_psnr:.2f}")
    print(f"SSIM: {final_ssim:.4f}")
    print(f"LPIPS: {final_lpips:.4f}")
    print(f"Saved to: {res_dir}")

    # Cleanup ASTRA
    operator.cleanup()


if __name__ == "__main__":
    run_experiment()
