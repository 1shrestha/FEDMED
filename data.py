"""
BraTS dataset loader for a single hospital node.

Each node is configured (via env var NODE_DATA_DIR) to point at ONLY its own
local slice of BraTS-formatted data. No node ever reads another node's
directory -- that boundary is what makes this cross-silo FL rather than a
centralized job, and it must be enforced at the infra/mount layer, not just
in code.
"""
import os
import glob
from typing import Tuple

import torch
from monai.data import Dataset, DataLoader, CacheDataset
from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    Orientationd,
    Spacingd,
    NormalizeIntensityd,
    RandCropByPosNegLabeld,
    RandFlipd,
    RandRotate90d,
    ConvertToMultiChannelBasedOnBratsClassesd,
    EnsureTyped,
)

MODALITIES = ["flair", "t1", "t1ce", "t2"]


def _list_cases(data_dir: str):
    cases = sorted(glob.glob(os.path.join(data_dir, "BraTS*")))
    items = []
    for case_dir in cases:
        case_id = os.path.basename(case_dir)
        images = [os.path.join(case_dir, f"{case_id}_{m}.nii.gz") for m in MODALITIES]
        label = os.path.join(case_dir, f"{case_id}_seg.nii.gz")
        if all(os.path.exists(p) for p in images) and os.path.exists(label):
            items.append({"image": images, "label": label})
    return items


def _transforms(train: bool):
    base = [
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys="image"),
        ConvertToMultiChannelBasedOnBratsClassesd(keys="label"),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        Spacingd(keys=["image", "label"], pixdim=(1.5, 1.5, 2.0),
                  mode=("bilinear", "nearest")),
        NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
    ]
    if train:
        base += [
            RandCropByPosNegLabeld(
                keys=["image", "label"], label_key="label",
                spatial_size=(96, 96, 64), pos=1, neg=1, num_samples=2,
            ),
            RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
            RandRotate90d(keys=["image", "label"], prob=0.5, max_k=3),
        ]
    base.append(EnsureTyped(keys=["image", "label"]))
    return Compose(base)


def get_dataloaders(node_data_dir: str, batch_size: int = 2,
                     val_split: float = 0.2, cache_rate: float = 0.5,
                     num_workers: int = 4) -> Tuple[DataLoader, DataLoader]:
    items = _list_cases(node_data_dir)
    if not items:
        raise RuntimeError(
            f"No BraTS cases found under {node_data_dir}. "
            "Mount this node's local data volume before starting training."
        )
    split = max(1, int(len(items) * (1 - val_split)))
    train_items, val_items = items[:split], items[split:] or items[:1]

    train_ds = CacheDataset(train_items, transform=_transforms(True), cache_rate=cache_rate)
    val_ds = Dataset(val_items, transform=_transforms(False))

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                               num_workers=num_workers, pin_memory=torch.cuda.is_available())
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=2)
    return train_loader, val_loader
