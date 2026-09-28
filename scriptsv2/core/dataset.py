from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from . import configuration as cfg

CLASS_NAMES = ["mitotic_figure", "not_mitotic_figure"]
PATCH_SIZE  = cfg.PATCH_SIZE


class CellsDataset(Dataset):
    """Dataset of cell patches with Z-score standardization.

    augment=True applies flips, rotation, color jitter, and blur.
    stats: tuple (mean, std) or (mean, std, data_range) to reuse already
    computed statistics. data_range = max_val - min_val over the standardized
    values, used as ssim_data_range.
    """

    def __init__(self, cells_dir: Path, augment: bool = False, flatten: bool = True,
                 stats: tuple = None):
        self.cells_dir = cells_dir
        self.flatten   = flatten
        self.augment   = augment

        self.paths  = []
        self.labels = []

        for label, cname in enumerate(CLASS_NAMES):
            class_dir = cells_dir / cname
            for p in sorted(class_dir.glob("*.png")):
                self.paths.append(p)
                self.labels.append(label)

        if len(self.paths) == 0:
            raise RuntimeError(f"No se encontraron PNGs en {cells_dir}")

        self.labels = torch.tensor(self.labels, dtype=torch.long)

        if stats is not None:
            if len(stats) == 3:
                self.mean, self.std, self.data_range = stats
            else:
                self.mean, self.std = stats
                self._compute_data_range_only()
        else:
            self._compute_statistics()

        self._build_transform()

    def _compute_statistics(self):
        print("Calculando estadísticas del dataset para estandarización...")
        pixel_sum    = 0.0
        pixel_sq_sum = 0.0
        num_pixels   = 0
        global_min   = float("inf")
        global_max   = float("-inf")

        for path in self.paths:
            arr = np.array(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0
            pixel_sum    += float(arr.sum())
            pixel_sq_sum += float((arr ** 2).sum())
            num_pixels   += arr.size
            global_min    = min(global_min, float(arr.min()))
            global_max    = max(global_max, float(arr.max()))

        mean = pixel_sum / num_pixels
        var  = max(pixel_sq_sum / num_pixels - mean ** 2, 0.0)
        std  = float(np.sqrt(var))

        if std < 1e-12:
            std = 1.0
            print("  [Aviso] std ~ 0 detectada. Se fuerza std=1.0.")

        self.mean = float(mean)
        self.std  = float(std)

        # data_range is computed over the already-standardized values
        z_min = (global_min - self.mean) / self.std
        z_max = (global_max - self.mean) / self.std
        self.data_range = float(z_max - z_min)

        print(f"  Media       : {self.mean:.6f}")
        print(f"  Std         : {self.std:.6f}")
        print(f"  Rango raw   : [{global_min:.4f}, {global_max:.4f}]")
        print(f"  Rango Z     : [{z_min:.4f}, {z_max:.4f}]")
        print(f"  data_range  : {self.data_range:.4f}  (usado para SSIM)")

    def _compute_data_range_only(self):
        """Compute data_range when stats=(mean, std) is provided externally."""
        global_min = float("inf")
        global_max = float("-inf")
        for path in self.paths:
            arr = np.array(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0
            global_min = min(global_min, float(arr.min()))
            global_max = max(global_max, float(arr.max()))
        z_min = (global_min - self.mean) / self.std
        z_max = (global_max - self.mean) / self.std
        self.data_range = float(z_max - z_min)

    def _build_transform(self):
        BI = transforms.InterpolationMode.BILINEAR

        if self.augment:
            # fill value in [0, 255] space using the dataset mean, to avoid extreme
            # pixel values in the corners after rotation
            fill_val = round(self.mean * 255)
            self._transform = transforms.Compose([
                transforms.RandomHorizontalFlip(p=cfg.AUG_HFLIP_P),
                transforms.RandomVerticalFlip(p=cfg.AUG_VFLIP_P),
                transforms.RandomRotation(cfg.AUG_ROTATION_DEGREES, interpolation=BI,
                                          fill=fill_val),
                transforms.ColorJitter(
                    brightness=cfg.AUG_BRIGHTNESS,
                    contrast=cfg.AUG_CONTRAST,
                ),
                transforms.RandomApply([
                    transforms.GaussianBlur(
                        kernel_size=cfg.AUG_BLUR_KERNEL,
                        sigma=(cfg.AUG_BLUR_SIGMA_MIN, cfg.AUG_BLUR_SIGMA_MAX),
                    )
                ], p=cfg.AUG_BLUR_P),
            ])
        else:
            self._transform = None

    def get_statistics(self):
        """Return (mean, std, data_range)."""
        return self.mean, self.std, self.data_range

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("RGB")

        if self._transform is not None:
            img = self._transform(img)

        arr = np.array(img, dtype=np.float32) / 255.0
        arr = (arr - self.mean) / self.std
        arr = arr.transpose(2, 0, 1)  # (H, W, C) -> (C, H, W)

        t = torch.from_numpy(arr.copy())

        if self.flatten:
            t = t.reshape(-1)

        return t, self.labels[idx]
