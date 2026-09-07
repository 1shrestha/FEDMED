"""
unet3d.py
---------
Shared 3D U-Net definition (MONAI). Both the centralized baseline and every
federated client import THIS SAME function, so architectures never drift
out of sync between nodes — a common, subtle bug in federated projects
(FedAvg silently breaks if client architectures don't match exactly).
"""

from monai.networks.nets import UNet
from monai.losses import DiceLoss
from monai.metrics import DiceMetric


def build_model(in_channels=4, out_channels=3):
    """
    in_channels=4  -> T1, T1ce, T2, FLAIR (standard BraTS modalities)
    out_channels=3 -> Whole Tumor (WT), Tumor Core (TC), Enhancing Tumor (ET)

    IMPORTANT: this UNet has 4 stride-2 downsampling steps, so input volumes
    (D, H, W) must each be divisible by 2^4 = 16, or the decoder's skip-connection
    concat will fail with a tensor size mismatch. Use volume sizes like 48, 64,
    80, 96... (real BraTS volumes are typically center-cropped/padded to 128^3
    or similar for exactly this reason).
    """
    model = UNet(
        spatial_dims=3,
        in_channels=in_channels,
        out_channels=out_channels,
        channels=(16, 32, 64, 128, 256),
        strides=(2, 2, 2, 2),
        num_res_units=2,
    )
    return model


def build_loss():
    # sigmoid=True because our labels are multi-channel binary masks
    # (a voxel can belong to multiple nested tumor regions at once)
    return DiceLoss(sigmoid=True)


def build_metric():
    return DiceMetric(include_background=True, reduction="mean")
