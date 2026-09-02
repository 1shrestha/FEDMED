"""
train_baseline.py  (WEEK 1 DELIVERABLE)
----------------------------------------
Trains a standard, centralized 3D U-Net on the (synthetic, or real BraTS)
dataset to establish the baseline Dice score. This number is your ceiling:
in Week 2 you'll show the federated model approaches it without any node
ever sharing raw data.

Run:
    python central_baseline/train_baseline.py --epochs 5 --volume-size 48

To switch to REAL BraTS later:
    1. Download a subset from the BraTS challenge (e.g. via the Synapse/Kaggle
       mirror of BraTS 2021) into ./data/BraTS2021/
    2. Replace `SyntheticBraTSDataset` below with a MONAI `Dataset` built from
       `monai.transforms` (LoadImaged, EnsureChannelFirstd, ScaleIntensityd,
       RandCropByPosNegLabeld) reading the .nii.gz files. The model code does
       not need to change at all — only the data loading.
"""

import argparse
import sys
import os
import torch
from torch.utils.data import DataLoader

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model.unet3d import build_model, build_loss, build_metric
from data.synthetic_data import SyntheticBraTSDataset


def train(epochs=5, volume_size=48, batch_size=2, n_samples=20, lr=1e-3, device=None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[baseline] device={device}")

    train_ds = SyntheticBraTSDataset(n_samples=n_samples, volume_size=volume_size, seed=0)
    val_ds = SyntheticBraTSDataset(n_samples=max(4, n_samples // 5), volume_size=volume_size, seed=999)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False)

    model = build_model().to(device)
    loss_fn = build_loss()
    metric_fn = build_metric()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    history = {"train_loss": [], "val_dice": []}

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        for batch in train_loader:
            images, labels = batch["image"].to(device), batch["label"].to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        epoch_loss /= len(train_loader)
        history["train_loss"].append(epoch_loss)

        # validation
        model.eval()
        metric_fn.reset()
        with torch.no_grad():
            for batch in val_loader:
                images, labels = batch["image"].to(device), batch["label"].to(device)
                outputs = torch.sigmoid(model(images))
                preds = (outputs > 0.5).float()
                metric_fn(y_pred=preds, y=labels)
        val_dice = metric_fn.aggregate().item()
        history["val_dice"].append(val_dice)

        print(f"[baseline] epoch {epoch}/{epochs}  loss={epoch_loss:.4f}  val_dice={val_dice:.4f}")

    torch.save(model.state_dict(), os.path.join(os.path.dirname(__file__), "baseline_model.pt"))
    print("[baseline] saved weights -> central_baseline/baseline_model.pt")
    print(f"[baseline] FINAL BASELINE DICE: {history['val_dice'][-1]:.4f}  <- this is your Week 1 headline number")
    return history


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--volume-size", type=int, default=48)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--n-samples", type=int, default=20)
    args = parser.parse_args()

    train(
        epochs=args.epochs,
        volume_size=args.volume_size,
        batch_size=args.batch_size,
        n_samples=args.n_samples,
    )
