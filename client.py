"""
FedMed client node. One instance of this runs inside each hospital's
network perimeter. It:
  1. Pulls the current global model weights from the server (plaintext --
     the *model* is not secret, only each hospital's *update* is).
  2. Trains locally on that hospital's private MRI data (never leaves
     the node).
  3. Computes the weight delta, encrypts it with the shared CKKS public
     context, and sends only ciphertext back to the server.

Run: NODE_ID=hospital-a NODE_DATA_DIR=/data/hospital-a python client.py
"""
import os
import sys
import logging

import numpy as np
import torch
import flwr as fl
from flwr.common import Parameters, FitRes, ndarrays_to_parameters, parameters_to_ndarrays
from monai.losses import DiceLoss
from monai.metrics import DiceMetric

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from client_node.model import build_unet3d
from client_node.data import get_dataloaders
from encryption import he_utils

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("fedmed.client")

NODE_ID = os.environ.get("NODE_ID", "hospital-a")
NODE_DATA_DIR = os.environ.get("NODE_DATA_DIR", f"/data/{NODE_ID}")
SERVER_ADDRESS = os.environ.get("SERVER_ADDRESS", "central-server:8080")
PUBLIC_CTX_PATH = os.environ.get("PUBLIC_CTX_PATH", "/keys/public_context.bin")
LOCAL_EPOCHS = int(os.environ.get("LOCAL_EPOCHS", "1"))
LR = float(os.environ.get("LEARNING_RATE", "1e-4"))
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class FedMedClient(fl.client.NumPyClient):
    def __init__(self):
        self.model = build_unet3d().to(DEVICE)
        self.train_loader, self.val_loader = get_dataloaders(NODE_DATA_DIR)
        self.loss_fn = DiceLoss(to_onehot_y=False, sigmoid=True)
        self.dice_metric = DiceMetric(include_background=True, reduction="mean")
        with open(PUBLIC_CTX_PATH, "rb") as f:
            self.public_ctx = he_utils.load_context(f.read())

    def get_parameters(self, config):
        return [p.detach().cpu().numpy() for p in self.model.parameters()]

    def set_parameters(self, parameters):
        for p, arr in zip(self.model.parameters(), parameters):
            p.data = torch.tensor(arr, dtype=p.dtype, device=DEVICE)

    def fit(self, parameters, config):
        before = [arr.copy() for arr in parameters]
        self.set_parameters(parameters)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=LR)
        self.model.train()
        running_loss, n_batches = 0.0, 0
        for _epoch in range(LOCAL_EPOCHS):
            for batch in self.train_loader:
                images = batch["image"].to(DEVICE)
                labels = batch["label"].to(DEVICE)
                optimizer.zero_grad()
                outputs = self.model(images)
                loss = self.loss_fn(outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
                n_batches += 1

        after = [p.detach().cpu().numpy() for p in self.model.parameters()]
        delta = [a - b for a, b in zip(after, before)]

        encrypted_delta = he_utils.encrypt_weights(self.public_ctx, delta)
        logger.info("node=%s round complete, avg_loss=%.4f, sending %d encrypted bytes",
                    NODE_ID, running_loss / max(n_batches, 1), len(encrypted_delta))

        # Ship the ciphertext as a single opaque ndarray of dtype uint8;
        # the custom server-side strategy knows to treat this specially.
        payload_arr = np.frombuffer(encrypted_delta, dtype=np.uint8).copy()
        metrics = {"node_id": NODE_ID, "loss": running_loss / max(n_batches, 1),
                   "encrypted": True}
        return [payload_arr], len(self.train_loader.dataset), metrics

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        self.model.eval()
        self.dice_metric.reset()
        val_loss = 0.0
        with torch.no_grad():
            for batch in self.val_loader:
                images = batch["image"].to(DEVICE)
                labels = batch["label"].to(DEVICE)
                outputs = self.model(images)
                val_loss += self.loss_fn(outputs, labels).item()
                preds = (torch.sigmoid(outputs) > 0.5).float()
                self.dice_metric(y_pred=preds, y=labels)
        dice = self.dice_metric.aggregate().item()
        n = max(len(self.val_loader), 1)
        return val_loss / n, len(self.val_loader.dataset), {"dice": dice, "node_id": NODE_ID}


if __name__ == "__main__":
    fl.client.start_numpy_client(server_address=SERVER_ADDRESS, client=FedMedClient())
