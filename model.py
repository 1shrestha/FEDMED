"""
3D U-Net for brain tumor MRI segmentation, built on MONAI.

Input:  4-channel MRI volume (FLAIR, T1, T1ce, T2), shape (B, 4, D, H, W)
Output: 3-channel segmentation mask (WT, TC, ET) per BraTS convention.
"""
from monai.networks.nets import UNet
from monai.networks.layers import Norm
import torch.nn as nn


def build_unet3d(in_channels: int = 4, out_channels: int = 3) -> nn.Module:
    """Standard MONAI 3D U-Net sized for single-GPU per-hospital training."""
    return UNet(
        spatial_dims=3,
        in_channels=in_channels,
        out_channels=out_channels,
        channels=(16, 32, 64, 128, 256),
        strides=(2, 2, 2, 2),
        num_res_units=2,
        norm=Norm.BATCH,
        dropout=0.1,
    )


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
