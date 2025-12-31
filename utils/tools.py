import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml


def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def save_config(config, path):
    with open(path, "w") as f:
        yaml.dump(config, f)


@torch.no_grad()
def sample_flow_matching(
    model, batch_size, img_size, device, steps=100, solver="euler"
):
    """
    Sample using Euler (fast, less accurate) or RK4 (slower, very smooth).
    """
    model.eval()
    x = torch.randn(batch_size, 1, img_size, img_size).to(device)
    dt = 1.0 / steps

    for i in range(steps):
        t_val = i / steps
        t = torch.full((batch_size,), t_val, device=device, dtype=torch.float32)

        if solver == "euler":
            v = model(x, t)
            x = x + v * dt

        elif solver == "rk4":
            # Runge-Kutta 4th Order Integration
            # k1
            v1 = model(x, t)

            # k2
            t2 = torch.full(
                (batch_size,), t_val + 0.5 * dt, device=device, dtype=torch.float32
            )
            x2 = x + 0.5 * dt * v1
            v2 = model(x2, t2)

            # k3
            x3 = x + 0.5 * dt * v2
            v3 = model(x3, t2)  # t is same as k2

            # k4
            t4 = torch.full(
                (batch_size,), t_val + dt, device=device, dtype=torch.float32
            )
            x4 = x + dt * v3
            v4 = model(x4, t4)

            # Combine
            x = x + (dt / 6.0) * (v1 + 2 * v2 + 2 * v3 + v4)

    model.train()
    return (x.clamp(-1, 1) + 1) / 2


def save_grid(images, path):
    # images: [B, 1, H, W]
    B = images.shape[0]
    grid_size = int(B**0.5)
    fig, axes = plt.subplots(grid_size, grid_size, figsize=(10, 10))
    if grid_size == 1:
        axes = np.array([axes])

    for i, ax in enumerate(axes.flat):
        if i < B:
            ax.imshow(images[i, 0].cpu().numpy(), cmap="gray")
            ax.axis("off")
        else:
            ax.axis("off")
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def seed_everything(seed=42):
    """
    Sets the seed for all random number generators to ensure reproducibility.
    """
    import random

    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.

    # Ensure deterministic behavior in CuDNN (slightly slower but consistent)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    print(f"[INFO] Global Seed set to {seed}")
