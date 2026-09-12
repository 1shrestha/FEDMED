"""
real_brats_loader.py
----------------------
Real BraTS dataset loader (replaces data/synthetic_data.py once you have
actual scans). Reads BraTS-style case folders:

    BraTS2021_00001/
        BraTS2021_00001_t1.nii.gz
        BraTS2021_00001_t1ce.nii.gz
        BraTS2021_00001_t2.nii.gz
        BraTS2021_00001_flair.nii.gz
        BraTS2021_00001_seg.nii.gz     <- label: 0=bg, 1=NCR, 2=ED, 4=ET

GETTING DATA (do this on your own machine — this sandbox has no network
access to Kaggle/Synapse):
  - Kaggle mirror: "BraTS2021" or "BraTS2020" dataset (search Kaggle for
    "brats 2021 task1"), download and unzip.
  - Or the official route: register at https://www.synapse.org for the
    BraTS challenge and download from there.
  - Either way, point --data-root at the folder containing the per-case
    subfolders above.

This module is self-tested below with SYNTHETIC .nii.gz files (real nibabel
files, real I/O path, fake voxels) so you can confirm the loader itself
works correctly before your real download finishes.
"""

import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset

from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, ScaleIntensityd,
    RandCropByPosNegLabeld, ConcatItemsd, DeleteItemsd, EnsureTyped,
)
from monai.data import Dataset as MonaiDataset

MODALITIES = ["t1", "t1ce", "t2", "flair"]


def find_case_dicts(data_root):
    """Scans data_root for BraTS case folders and builds MONAI-style dict entries."""
    cases = []
    for case_dir in sorted(glob.glob(os.path.join(data_root, "*"))):
        if not os.path.isdir(case_dir):
            continue
        case_id = os.path.basename(case_dir)
        entry = {mod: os.path.join(case_dir, f"{case_id}_{mod}.nii.gz") for mod in MODALITIES}
        entry["seg"] = os.path.join(case_dir, f"{case_id}_seg.nii.gz")
        if all(os.path.exists(p) for p in entry.values()):
            cases.append(entry)
    return cases


def _remap_labels(d):
    """
    BraTS raw labels are {0, 1, 2, 4}. Converts to the same 3-channel
    nested-region format (WT, TC, ET) used by data/synthetic_data.py and
    model/unet3d.py, so NOTHING else in the pipeline needs to change.
    """
    seg = d["seg"]  # shape [1, D, H, W], values in {0,1,2,4}
    wt = (seg > 0).float()
    tc = ((seg == 1) | (seg == 4)).float()
    et = (seg == 4).float()
    d["label"] = torch.cat([wt, tc, et], dim=0)
    del d["seg"]
    return d


def build_transforms(patch_size=(64, 64, 64)):
    return Compose([
        LoadImaged(keys=MODALITIES + ["seg"]),
        EnsureChannelFirstd(keys=MODALITIES + ["seg"]),
        ScaleIntensityd(keys=MODALITIES),
        ConcatItemsd(keys=MODALITIES, name="image"),
        DeleteItemsd(keys=MODALITIES),
        RandCropByPosNegLabeld(
            keys=["image", "seg"], label_key="seg", spatial_size=patch_size,
            pos=1, neg=1, num_samples=1,
        ),
        EnsureTyped(keys=["image", "seg"]),
    ])


class BraTSRealDataset(Dataset):
    """Thin wrapper so the __getitem__ output matches SyntheticBraTSDataset exactly:
    {"image": [4,D,H,W], "label": [3,D,H,W]} — drop-in replacement."""

    def __init__(self, data_root, patch_size=(64, 64, 64)):
        self.cases = find_case_dicts(data_root)
        if not self.cases:
            raise FileNotFoundError(
                f"No complete BraTS cases found under {data_root}. "
                f"Expected subfolders each containing *_t1.nii.gz, *_t1ce.nii.gz, "
                f"*_t2.nii.gz, *_flair.nii.gz, *_seg.nii.gz"
            )
        transform = build_transforms(patch_size)
        self._monai_ds = MonaiDataset(data=self.cases, transform=transform)

    def __len__(self):
        return len(self._monai_ds)

    def __getitem__(self, idx):
        item = self._monai_ds[idx][0]  # RandCropByPosNegLabeld returns a list of 1
        item = _remap_labels(item)
        return {"image": item["image"], "label": item["label"]}


def partition_real_dataset(data_root, n_clients=3, patch_size=(64, 64, 64)):
    """Splits the real case list evenly across hospital nodes — same signature
    shape as data.synthetic_data.partition_dataset, for drop-in use in
    federated/client.py."""
    full = BraTSRealDataset(data_root, patch_size)
    n = len(full.cases)
    per_client = n // n_clients
    subsets = []
    for i in range(n_clients):
        start, end = i * per_client, (i + 1) * per_client if i < n_clients - 1 else n
        client_cases = full.cases[start:end]
        ds = BraTSRealDataset.__new__(BraTSRealDataset)  # skip __init__'s glob rescan
        ds.cases = client_cases
        ds._monai_ds = MonaiDataset(data=client_cases, transform=build_transforms(patch_size))
        subsets.append(ds)
    return subsets


# ----------------------------------------------------------------------------
# Self-test with SYNTHETIC .nii.gz files — proves the LOADER (not the data)
# is correct: real file I/O, real MONAI transform pipeline, real label
# remapping. Run this once before your real download to catch path/shape
# bugs cheaply.
# ----------------------------------------------------------------------------
def _make_fake_case(root, case_id, size=48):
    import nibabel as nib
    case_dir = os.path.join(root, case_id)
    os.makedirs(case_dir, exist_ok=True)
    affine = np.eye(4)
    for mod in MODALITIES:
        vol = np.random.rand(size, size, size).astype(np.float32)
        nib.save(nib.Nifti1Image(vol, affine), os.path.join(case_dir, f"{case_id}_{mod}.nii.gz"))
    seg = np.zeros((size, size, size), dtype=np.uint8)
    c = size // 2
    seg[c - 8:c + 8, c - 8:c + 8, c - 8:c + 8] = 1
    seg[c - 4:c + 4, c - 4:c + 4, c - 4:c + 4] = 4
    nib.save(nib.Nifti1Image(seg, affine), os.path.join(case_dir, f"{case_id}_seg.nii.gz"))


if __name__ == "__main__":
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        for i in range(3):
            _make_fake_case(tmp, f"FAKE_{i:03d}")
        ds = BraTSRealDataset(tmp, patch_size=(32, 32, 32))
        print(f"found {len(ds)} case(s)")
        sample = ds[0]
        print("image shape:", sample["image"].shape, "label shape:", sample["label"].shape)
        print("PASS — real BraTS loader pipeline (I/O + transforms + label remap) works end to end")
