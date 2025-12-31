import argparse
import os
import time

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from data.dataset import MayoDataset
from models.unet import UNet
from utils.tools import load_config, sample_flow_matching, save_config, save_grid


def get_params_count(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def train():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    parser.add_argument(
        "--resume", type=str, default=None, help="Path to .pth file to resume from"
    )
    args = parser.parse_args()

    # 1. Load Configuration
    print("--- Flow Matching Training Setup ---")
    config = load_config(args.config)
    device = torch.device(config["device"])
    print(f"[INFO] Experiment Name: {config['experiment_name']}")
    print(f"[INFO] Device: {device}")

    # 2. Setup Folders
    exp_dir = os.path.join("outputs", config["experiment_name"])
    weights_dir = os.path.join(exp_dir, "weights")
    samples_dir = os.path.join(exp_dir, "samples")

    os.makedirs(weights_dir, exist_ok=True)
    os.makedirs(samples_dir, exist_ok=True)

    # Save config for reproducibility
    save_config(config, os.path.join(weights_dir, "config.yaml"))
    print(f"[INFO] Directories created.")
    print(f"       Weights: {weights_dir}")
    print(f"       Samples: {samples_dir}")

    # 3. Data Loading
    print(f"\n[INFO] Initializing dataset...")
    data_dir = config["data"]["data_dir"]
    dataset = MayoDataset(
        data_dir, phase="train", img_size=config["data"]["img_size"], config=config
    )

    # Feedback on Data
    if len(dataset) == 0:
        print(
            f"[ERROR] No images found in {os.path.join(data_dir, 'train')}. Check your path!"
        )
        return

    print(f"[INFO] Found {len(dataset)} images in '{data_dir}'.")
    print(f"[INFO] Data Augmentation: Enabled (ElasticTransform, Flip, Rotate)")

    dataloader = DataLoader(
        dataset,
        batch_size=config["data"]["batch_size"],
        shuffle=True,
        num_workers=config["data"]["num_workers"],
        pin_memory=True if device.type == "cuda" else False,
    )
    print(f"[INFO] DataLoader ready. Batch size: {config['data']['batch_size']}")

    # 4. Model & Optimizer
    print(f"\n[INFO] Building U-Net model...")
    model = UNet(config).to(device)

    param_count = get_params_count(model)
    print(
        f"[INFO] Model loaded successfully. Trainable parameters: {param_count/1e6:.2f} M"
    )

    # --- Resume Logic ---
    if args.resume:
        print(f"[INFO] Resuming/Fine-tuning from: {args.resume}")
        if os.path.isfile(args.resume):
            checkpoint = torch.load(args.resume, map_location=device)

            # Load weights (ignoring mismatch keys if you changed architecture slightly)
            # strictly=True ensures we crash if layers don't match, which is good for safety here
            model.load_state_dict(checkpoint, strict=True)
            print("[INFO] Weights loaded successfully.")
        else:
            print(f"[ERROR] No weight file found at {args.resume}")
            return

    optimizer = optim.AdamW(model.parameters(), lr=config["training"]["learning_rate"])
    sigma_min = config["training"]["sigma_min"]

    # 5. Training Loop
    epochs = config["training"]["epochs"]
    print(f"\n--- Starting Training for {epochs} Epochs ---")

    start_time = time.time()

    for epoch in range(1, epochs + 1):
        model.train()
        pbar = tqdm(dataloader, desc=f"Epoch {epoch}/{epochs}", leave=True)
        avg_loss = 0.0

        for x1 in pbar:
            x1 = x1.to(device)  # Target data (x_1)
            b = x1.shape[0]

            # --- Flow Matching Training Step ---

            # 1. Sample time t ~ Uniform(0, 1)
            t = torch.rand(b, device=device).float()

            # 2. Sample noise x_0 ~ Normal(0, 1)
            x0 = torch.randn_like(x1)

            # 3. Compute x_t (Interpolation path)
            # We use Optimal Transport path: x_t = (1 - (1 - sigma_min) * t) * x0 + t * x1
            t_reshaped = t.view(b, 1, 1, 1)
            x_t = (1 - (1 - sigma_min) * t_reshaped) * x0 + t_reshaped * x1

            # 4. Compute target velocity u_t
            # u_t = dx_t/dt = x1 - (1 - sigma_min) * x0
            u_t = x1 - (1 - sigma_min) * x0

            # 5. Predict velocity field v_t
            v_pred = model(x_t, t)

            # 6. Loss (MSE)
            loss = torch.mean((v_pred - u_t) ** 2)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            avg_loss += loss.item()
            pbar.set_postfix({"Loss": f"{loss.item():.4f}"})

        # Calculate average loss for the epoch
        avg_loss /= len(dataloader)

        # --- Feedback & Saving ---
        if epoch % config["training"]["sample_every"] == 0:
            # Simple print to break the tqdm line and show activity
            print(f"\n[INFO] Epoch {epoch}: Generating samples...")
            images = sample_flow_matching(model, 4, config["data"]["img_size"], device)
            save_path = os.path.join(samples_dir, f"epoch_{epoch}.png")
            save_grid(images, save_path)
            print(f"[INFO] Samples saved to {save_path}")

        if epoch % config["training"]["save_every"] == 0:
            save_path = os.path.join(weights_dir, f"model_epoch_{epoch}.pth")
            torch.save(model.state_dict(), save_path)
            torch.save(model.state_dict(), os.path.join(weights_dir, "latest.pth"))
            print(f"[INFO] Checkpoint saved: {save_path}")

    total_time = time.time() - start_time
    print(f"\n--- Training Complete in {total_time/60:.2f} minutes ---")
    print(f"Best weights can be found in: {weights_dir}")


if __name__ == "__main__":
    train()
