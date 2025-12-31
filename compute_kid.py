import argparse
import glob
import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from torchmetrics.image.kid import KernelInceptionDistance
from tqdm import tqdm

# Project imports
from models.unet import UNet
from utils.tools import load_config


def parse_args():
    parser = argparse.ArgumentParser(description="Compute KID score vs Sampling Steps")
    parser.add_argument(
        "--config", type=str, default="configs/config.yaml", help="Path to config file"
    )
    parser.add_argument(
        "--weights", type=str, required=True, help="Path to model weights .pth"
    )
    parser.add_argument(
        "--test_dir",
        type=str,
        default="../../data/Mayo/test",
        help="Path to test set folder",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size for generation and metric",
    )
    parser.add_argument(
        "--num_gen",
        type=int,
        default=500,
        help="Number of samples to generate for KID calculation",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=5,
        help="Number of times to repeat calculation per step count",
    )
    # The steps to evaluate. You can change this list default or pass it via command line if you extended this logic
    parser.add_argument(
        "--steps_list",
        type=int,
        nargs="+",
        default=[10, 20, 50, 100],
        help="List of step counts to evaluate",
    )
    parser.add_argument(
        "--save_path",
        type=str,
        default="experiments/kid_plot.png",
        help="Path to save the final plot",
    )
    return parser.parse_args()


def load_real_data(test_dir, img_size):
    """
    Recursively finds all png files in test_dir, loads them, resizes,
    and returns a uint8 tensor (N, 3, H, W).
    """
    print(f"[INFO] Loading real images from {test_dir}...")
    files = glob.glob(os.path.join(test_dir, "**", "*.png"), recursive=True)

    if len(files) == 0:
        raise ValueError(f"No images found in {test_dir}")

    images = []
    for f in tqdm(files, desc="Loading Real Data"):
        img = cv2.imread(f, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue

        img = cv2.resize(img, (img_size, img_size))
        # KID expects (N, 3, H, W) in uint8 [0, 255]
        # We duplicate the grayscale channel to 3 channels
        img = np.stack([img, img, img], axis=-1)
        images.append(img)

    # Convert to Tensor (N, 3, H, W)
    data = torch.from_numpy(np.array(images)).permute(0, 3, 1, 2)
    print(f"[INFO] Loaded {len(data)} real images.")
    return data


@torch.no_grad()
def generate_fake_data(model, num_samples, batch_size, img_size, steps, device):
    """
    Generates 'num_samples' images using the model with 'steps' integration steps.
    Returns uint8 tensor (N, 3, H, W).
    """
    model.eval()
    fake_images = []
    num_batches = (num_samples + batch_size - 1) // batch_size

    for _ in range(num_batches):
        current_bs = min(batch_size, num_samples - len(fake_images))

        # --- Sampling Loop (Euler) ---
        x = torch.randn(current_bs, 1, img_size, img_size).to(device)
        dt = 1.0 / steps

        for i in range(steps):
            t = torch.full((current_bs,), i / steps, device=device).float()
            v = model(x, t)
            x = x + v * dt

        # Post-process: [-1, 1] -> [0, 255] uint8
        x = (x.clamp(-1, 1) + 1) / 2 * 255
        x = x.to(torch.uint8).cpu()

        # Repeat channels 1 -> 3
        x = x.repeat(1, 3, 1, 1)
        fake_images.append(x)

    return torch.cat(fake_images, dim=0)[:num_samples]


def main():
    args = parse_args()
    config = load_config(args.config)
    device = torch.device(config["device"])
    img_size = config["data"]["img_size"]

    # 1. Load Model
    print(f"[INFO] Loading Model weights from {args.weights}...")
    model = UNet(config).to(device)
    model.load_state_dict(torch.load(args.weights, map_location=device))

    # 2. Load Real Data
    # We load this once. It stays in CPU RAM until needed to save GPU memory.
    real_data = load_real_data(args.test_dir, img_size)

    # 3. Setup KID Metric
    # subset_size determines how many subsets are used for the KID estimation.
    # Usually 50-100 is standard.
    kid_metric = KernelInceptionDistance(subset_size=50).to(device)

    results_mean = []
    results_std = []

    print("\n--- Starting KID Evaluation ---")
    print(f"Steps to evaluate: {args.steps_list}")
    print(f"Repetitions per step: {args.repeats}")

    # 4. Evaluation Loop
    for steps in args.steps_list:
        step_scores = []

        print(f"\nProcessing: {steps} Steps")

        for r in range(args.repeats):
            # Reset metric for this run
            kid_metric.reset()

            # A. Feed Real Data (in batches to avoid OOM)
            # We shuffle real data to get random subsets if num_gen < len(real_data)
            # or just to be robust.
            perm = torch.randperm(len(real_data))
            curr_real = real_data[perm[: args.num_gen]]

            # Feed real images to KID (needs to be on GPU)
            # We assume args.num_gen fits in GPU memory if batched,
            # otherwise loop this update too.
            dataset_real = TensorDataset(curr_real)
            loader_real = DataLoader(dataset_real, batch_size=args.batch_size)

            for (batch,) in loader_real:
                kid_metric.update(batch.to(device), real=True)

            # B. Generate & Feed Fake Data
            fake_data = generate_fake_data(
                model, args.num_gen, args.batch_size, img_size, steps, device
            )

            dataset_fake = TensorDataset(fake_data)
            loader_fake = DataLoader(dataset_fake, batch_size=args.batch_size)

            for (batch,) in loader_fake:
                kid_metric.update(batch.to(device), real=False)

            # C. Compute
            score = kid_metric.compute()
            kid_val = score[
                0
            ].item()  # tuple (mean, std) returned by metric, we take mean of calculation
            step_scores.append(kid_val)

            print(f"  Rep {r+1}/{args.repeats}: KID = {kid_val:.4f}")

        # Calculate stats for this step count
        mu = np.mean(step_scores)
        std = np.std(step_scores)
        results_mean.append(mu)
        results_std.append(std)
        print(f"  >> Result for {steps} steps: {mu:.4f} ± {std:.4f}")

    # 5. Plotting
    print(f"\n[INFO] Plotting results to {args.save_path}...")
    steps_arr = np.array(args.steps_list)
    means_arr = np.array(results_mean)
    stds_arr = np.array(results_std)

    plt.figure(figsize=(10, 6))
    plt.plot(steps_arr, means_arr, marker="o", label="Mean KID", color="blue")
    plt.fill_between(
        steps_arr,
        means_arr - stds_arr,
        means_arr + stds_arr,
        color="blue",
        alpha=0.2,
        label="Standard Deviation",
    )

    plt.title("KID Score vs. Sampling Steps (Flow Matching)")
    plt.xlabel("Number of Sampling Steps")
    plt.ylabel("KID Score (Lower is Better)")
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.xticks(steps_arr)
    plt.legend()

    # Ensure directory exists
    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
    plt.savefig(args.save_path)
    print("Done.")


if __name__ == "__main__":
    main()
