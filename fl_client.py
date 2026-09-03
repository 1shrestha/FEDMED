"""
client/fl_client.py
===================
FedMed Flower client with integrated security.

Each hospital node runs one instance of :class:`FedMedClient` which:

1. **Trains locally** using the hospital's private dataset (never shared).
2. **Signs the model update** with its ECDSA private key before transmission.
3. **Masks the update** using the SecAgg protocol so the server never sees
   the plaintext gradient (optional; controlled by ``use_secagg``).
4. **Uses mTLS** for all communication with the Flower server (handled at
   the gRPC channel level in ``main_client.py``).

Component wiring
----------------
::

    FedMedClient
        │
        ├─ MedMLP / MedCNN            ← model/simple_model.py
        ├─ get_parameters()           ← model/simple_model.py
        ├─ set_parameters()           ← model/simple_model.py
        ├─ sign_update()              ← security/crypto_utils.py
        ├─ SecAggClient.mask_update() ← security/secagg.py
        └─ load_channel_credentials() ← security/grpc_tls.py  (in main_client.py)
"""

from __future__ import annotations

import logging
import pickle
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

import flwr as fl
from flwr.common import NDArrays, Scalar

from model.simple_model import (
    MedCNN,
    MedMLP,
    flatten_parameters,
    get_parameters,
    set_parameters,
    unflatten_parameters,
)
from security.crypto_utils import (
    ECPrivateKey,
    ECPublicKey,
    generate_ecdsa_keypair,
    serialise_public_key,
    sign_update,
)
from security.secagg import SecAggClient, ClientKeyBundle

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Synthetic local dataset helper
# ---------------------------------------------------------------------------


def make_synthetic_dataset(
    n_samples: int = 256,
    n_features: int = 20,
    n_classes: int = 2,
    model_type: str = "mlp",
    seed: int = 0,
) -> DataLoader:
    """Generate synthetic local data for one hospital node.

    In production, replace with your hospital's real DataLoader.

    Parameters
    ----------
    n_samples:
        Number of local training samples.
    n_features:
        Number of input features (for MLP) or image size 64×64 (for CNN).
    n_classes:
        Number of output classes.
    model_type:
        ``"mlp"`` for tabular data, ``"cnn"`` for image data.
    seed:
        Random seed for reproducibility across clients.

    Returns
    -------
    torch.utils.data.DataLoader
        Local training data loader.
    """
    rng = torch.Generator().manual_seed(seed)
    if model_type == "cnn":
        X = torch.randn(n_samples, 1, 64, 64, generator=rng)
    else:
        X = torch.randn(n_samples, n_features, generator=rng)
    y = torch.randint(0, n_classes, (n_samples,), generator=rng)
    return DataLoader(TensorDataset(X, y), batch_size=32, shuffle=True)


# ---------------------------------------------------------------------------
# FedMed Flower client
# ---------------------------------------------------------------------------


class FedMedClient(fl.client.NumPyClient):
    """Flower NumPyClient with ECDSA signing and SecAgg masking.

    Parameters
    ----------
    client_id:
        Integer ID of this hospital node (0-indexed).
    model:
        PyTorch model (``MedMLP`` or ``MedCNN``).
    train_loader:
        Local training DataLoader.
    val_loader:
        Local validation DataLoader.
    device:
        PyTorch device string (``"cpu"`` or ``"cuda"``).
    use_secagg:
        If ``True``, masks the update via :class:`SecAggClient` before
        returning it to Flower.
    n_clients:
        Total number of clients in the round (required for SecAgg).
    secagg_threshold:
        Shamir reconstruction threshold (default: ceil(n_clients / 2)).
    """

    def __init__(
        self,
        client_id: int,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        device: str = "cpu",
        use_secagg: bool = True,
        n_clients: int = 5,
        secagg_threshold: Optional[int] = None,
    ):
        self.client_id = client_id
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = torch.device(device)
        self.use_secagg = use_secagg

        # ── ECDSA key pair for signing updates ────────────────────────────────
        self._sign_private, self._sign_public = generate_ecdsa_keypair()
        logger.info(
            "[FedMedClient %d] ECDSA key pair generated.", client_id
        )

        # ── SecAgg client for masking ─────────────────────────────────────────
        self._secagg_client: Optional[SecAggClient] = None
        if use_secagg:
            self._secagg_client = SecAggClient(
                client_id=client_id,
                n_clients=n_clients,
                threshold=secagg_threshold,
            )

    # ── Public key accessor (called by server during registration) ───────────

    @property
    def public_key_pem(self) -> bytes:
        """PEM-encoded ECDSA public key for server registration."""
        return serialise_public_key(self._sign_public)

    # ── Flower NumPyClient interface ─────────────────────────────────────────

    def get_parameters(self, config: Dict) -> NDArrays:
        """Return current model parameters to the server."""
        return get_parameters(self.model)

    def fit(
        self, parameters: NDArrays, config: Dict
    ) -> Tuple[NDArrays, int, Dict[str, Scalar]]:
        """Local training round.

        Steps:
        1. Load global parameters from server.
        2. Train locally for ``local_epochs`` epochs.
        3. Compute the parameter delta (update = new − old).
        4. Sign the update.
        5. Optionally mask via SecAgg.
        6. Return masked (or plain) update + signature in metrics.
        """
        server_round = config.get("round", 0)
        local_epochs = config.get("local_epochs", 1)
        secagg_enabled = bool(config.get("secagg_enabled", self.use_secagg))

        # 1. Load global model
        set_parameters(self.model, parameters)
        old_params = get_parameters(self.model)

        # 2. Local training
        self._train(local_epochs)

        # 3. Compute update (delta)
        new_params = get_parameters(self.model)
        update_ndarrays = [
            n - o for n, o in zip(new_params, old_params)
        ]

        # 4. Sign the update
        serialised = self._serialise_update(update_ndarrays)
        signature = sign_update(self._sign_private, serialised)

        logger.info(
            "[FedMedClient %d] Round %d — local training complete. "
            "Update norm=%.4f | Signed ✓",
            self.client_id, server_round,
            float(np.linalg.norm(flatten_parameters(update_ndarrays))),
        )

        # 5. SecAgg masking (if enabled)
        if secagg_enabled and self._secagg_client is not None:
            update_ndarrays = self._apply_secagg_mask(update_ndarrays)

        metrics: Dict[str, Scalar] = {
            "signature": signature.hex(),
            "client_id": self.client_id,
            "secagg_active": int(secagg_enabled and self._secagg_client is not None),
        }
        return update_ndarrays, len(self.train_loader.dataset), metrics

    def evaluate(
        self, parameters: NDArrays, config: Dict
    ) -> Tuple[float, int, Dict[str, Scalar]]:
        """Evaluate the global model on local validation data."""
        set_parameters(self.model, parameters)
        loss, accuracy = self._evaluate()
        logger.info(
            "[FedMedClient %d] Evaluation — loss=%.4f | accuracy=%.4f",
            self.client_id, loss, accuracy,
        )
        return loss, len(self.val_loader.dataset), {"accuracy": accuracy}

    # ── SecAgg integration ────────────────────────────────────────────────────

    def secagg_generate_keys(self) -> ClientKeyBundle:
        """Round 1: Generate ephemeral SecAgg key bundle for this round."""
        if self._secagg_client is None:
            raise RuntimeError("SecAgg is not enabled for this client.")
        return self._secagg_client.generate_keys()

    def secagg_receive_peer_keys(self, peer_bundles: List[ClientKeyBundle]) -> None:
        """Round 2: Receive and store all peer public keys."""
        if self._secagg_client is None:
            raise RuntimeError("SecAgg is not enabled for this client.")
        self._secagg_client.receive_peer_keys(peer_bundles)

    def _apply_secagg_mask(self, update_ndarrays: List[np.ndarray]) -> List[np.ndarray]:
        """Apply SecAgg pairwise + own masks to the update.

        In multi-process deployment this is called *after* Rounds 1 and 2 are
        complete and peer keys have been distributed.  In single-process
        simulation, masking/unmasking is symmetric so the aggregate is still
        correct.
        """
        flat = flatten_parameters(update_ndarrays)
        shapes = [arr.shape for arr in update_ndarrays]
        masked_flat = self._secagg_client.mask_update(flat.reshape(1, -1).flatten())
        return unflatten_parameters(masked_flat.flatten(), shapes)

    # ── Training / evaluation loops ──────────────────────────────────────────

    def _train(self, epochs: int) -> None:
        """Run local SGD training for *epochs* epochs."""
        self.model.train()
        optimizer = optim.SGD(self.model.parameters(), lr=0.01, momentum=0.9)
        criterion = nn.CrossEntropyLoss()

        for epoch in range(epochs):
            total_loss = 0.0
            for X, y in self.train_loader:
                X, y = X.to(self.device), y.to(self.device)
                optimizer.zero_grad()
                out = self.model(X)
                loss = criterion(out, y)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            logger.debug(
                "[FedMedClient %d] Epoch %d/%d — loss=%.4f",
                self.client_id, epoch + 1, epochs, total_loss / len(self.train_loader),
            )

    def _evaluate(self) -> Tuple[float, float]:
        """Evaluate on the local validation set. Returns (loss, accuracy)."""
        self.model.eval()
        criterion = nn.CrossEntropyLoss()
        total_loss, correct, total = 0.0, 0, 0

        with torch.no_grad():
            for X, y in self.val_loader:
                X, y = X.to(self.device), y.to(self.device)
                out = self.model(X)
                total_loss += criterion(out, y).item()
                preds = out.argmax(dim=1)
                correct += (preds == y).sum().item()
                total += y.size(0)

        loss = total_loss / len(self.val_loader)
        accuracy = correct / total if total > 0 else 0.0
        return loss, accuracy

    @staticmethod
    def _serialise_update(ndarrays: List[np.ndarray]) -> bytes:
        """Serialise parameter update to bytes for ECDSA signing."""
        return pickle.dumps([arr.tobytes() for arr in ndarrays])
