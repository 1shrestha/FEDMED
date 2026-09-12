"""
synthetic_data.py
------------------
Generates small synthetic 4-modality MRI volumes + tumor segmentation masks
that mimic the shape/structure of BraTS data (T1, T1ce, T2, FLAIR -> 3 tumor
sub-region labels). This lets you build and debug the ENTIRE pipeline
(model, federated loop, encryption, dashboard) before you plug in real BraTS
data, which is large (~few GB) and requires registration to download.

Swap this out for a real BraTS loader once the pipeline works end-to-end.
Real BraTS loading references are in central_baseline/train_baseline.py
as comments.
"""

import numpy as np
import torch
from torch.utils.data import Dataset


def _make_blob(shape, center, radius, value=1.0):
    """Paints a soft spherical blob into a 3D volume — stands in for a tumor."""
    zz, yy, xx = np.ogrid[:shape[0], :shape[1], :shape[2]]
    dist = np.sqrt((zz - center[0]) ** 2 + (yy - center[1]) ** 2 + (xx - center[2]) ** 2)
    return (dist <= radius).astype(np.float32) * value


class SyntheticBraTSDataset(Dataset):
    """
    Produces (image, label) pairs shaped like BraTS:
      image: [4, D, H, W]  (T1, T1ce, T2, FLAIR)
      label: [3, D, H, W]  (multi-channel one-hot: WT, TC, ET tumor regions)

    n_samples: how many synthetic "patients" to generate
    volume_size: cube edge length (keep small, e.g. 64, for fast local dev;
                 real BraTS volumes are 240x240x155 and need patch-cropping)
    seed: for reproducibility across the 3 simulated hospital nodes
    """

    def __init__(self, n_samples=20, volume_size=64, seed=0):
        self.n_samples = n_samples
        self.volume_size = volume_size
        self.rng = np.random.RandomState(seed)

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        s = self.volume_size
        rng = np.random.RandomState(self.rng.randint(0, 1_000_000) + idx)

        # base tissue signal per modality (slightly different intensity stats
        # per modality, like real MRI sequences do)
        image = np.stack([
            rng.normal(0.5, 0.1, (s, s, s)),   # T1
            rng.normal(0.4, 0.1, (s, s, s)),   # T1ce
            rng.normal(0.6, 0.1, (s, s, s)),   # T2
            rng.normal(0.55, 0.1, (s, s, s)),  # FLAIR
        ]).astype(np.float32)

        # random tumor location/size -> nested regions like real BraTS labels
        center = rng.randint(s // 4, 3 * s // 4, size=3)
        whole_tumor = _make_blob((s, s, s), center, radius=rng.randint(8, 14))
        tumor_core = _make_blob((s, s, s), center, radius=rng.randint(4, 7))
        enhancing_tumor = _make_blob((s, s, s), center, radius=rng.randint(2, 4))

        # tumor regions bump signal intensity in the "lesion-sensitive" modalities
        image[1] += whole_tumor * 0.3  # T1ce
        image[3] += whole_tumor * 0.3  # FLAIR

        label = np.stack([whole_tumor, tumor_core, enhancing_tumor]).astype(np.float32)

        return {
            "image": torch.from_numpy(image),
            "label": torch.from_numpy(label),
        }


def partition_dataset(n_total=60, n_clients=3, volume_size=64, iid=True):
    """
    Splits a synthetic dataset across n_clients "hospitals".
    iid=True  -> each hospital gets a random, similarly-distributed slice
    iid=False -> each hospital gets a distinct tumor-size bias (simulates
                 real cross-silo heterogeneity: different scanner/population
                 statistics per hospital, a core challenge in federated
                 medical learning worth discussing in your report)
    """
    per_client = n_total // n_clients
    datasets = []
    for i in range(n_clients):
        seed = i if iid else i * 97 + 13  # different seed streams => non-IID skew
        datasets.append(SyntheticBraTSDataset(n_samples=per_client, volume_size=volume_size, seed=seed))
    return datasets


if __name__ == "__main__":
    ds = SyntheticBraTSDataset(n_samples=2, volume_size=32)
    sample = ds[0]
    print("image shape:", sample["image"].shape)
    print("label shape:", sample["label"].shape)
