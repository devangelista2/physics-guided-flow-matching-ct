import torch
import torch.nn as nn
from generative.networks.nets import DiffusionModelUNet


def create_monai_unet(config):
    """
    Creates the MONAI DiffusionModelUNet based on provided config.
    """
    return DiffusionModelUNet(
        spatial_dims=2,
        in_channels=config.get("in_channels", 1),
        out_channels=config.get("out_channels", 1),
        num_channels=config.get("block_out_channels", (64, 128, 256)),
        attention_levels=config.get("attention_levels", (False, True, True)),
        num_res_blocks=config.get("layers_per_block", 2),
        num_head_channels=config.get("num_head_channels", 32),
    )


def load_diffusion_model(model_path, config, device="cpu"):
    """
    Loads weights from a .pth file into the MONAI model.
    """
    print(f"Loading Diffusion Model from {model_path}...")
    model = create_monai_unet(config)

    state_dict = torch.load(model_path, map_location=device)

    # Robust loading: Handle 'module.' prefix if saved with DataParallel
    new_state_dict = {}
    for k, v in state_dict.items():
        name = k[7:] if k.startswith("module.") else k
        new_state_dict[name] = v

    # Loose loading to ignore potential header mismatches
    missing, unexpected = model.load_state_dict(new_state_dict, strict=False)
    if missing:
        print(f"[Warn] Missing keys: {len(missing)}")

    model.to(device)
    model.eval()
    return model
