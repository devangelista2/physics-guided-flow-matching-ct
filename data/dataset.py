import glob
import os

import albumentations as A
import cv2
import numpy as np
import torch
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset


class MayoDataset(Dataset):
    def __init__(self, root_dir, phase="train", img_size=256, config=None):
        self.files = glob.glob(os.path.join(root_dir, phase, "*", "*.png"))
        self.img_size = img_size
        self.phase = phase

        if phase == "train" and config:
            self.transform = A.Compose(
                [
                    A.Resize(img_size, img_size),
                    A.ElasticTransform(
                        alpha=config["data"]["elastic_alpha"],
                        sigma=config["data"]["elastic_sigma"],
                        p=config["data"]["aug_prob"],
                        border_mode=cv2.BORDER_CONSTANT,
                        value=0,
                    ),
                    A.HorizontalFlip(p=0.5),
                    A.Rotate(limit=15, p=0.5),
                    A.Normalize(mean=(0.5,), std=(0.5,)),  # Normalize to [-1, 1]
                    ToTensorV2(),
                ]
            )
        else:
            self.transform = A.Compose(
                [
                    A.Resize(img_size, img_size),
                    A.Normalize(mean=(0.5,), std=(0.5,)),
                    ToTensorV2(),
                ]
            )

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img_path = self.files[idx]
        # Read as grayscale
        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

        if self.transform:
            augmented = self.transform(image=image)
            image = augmented["image"]

        return image
