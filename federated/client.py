"""
client.py  (WEEK 1-4 DELIVERABLE — this file grows across the project)
------------------------------------------------------------------------
Represents ONE hospital node. Run three separate instances of this process
(different --client-id, same --server-address) to simulate the 3-hospital
cross-silo setup. In Week 1-2 this is plaintext FedAvg. Flip --encrypt in
Week 3 to add homomorphic encryption. Flip --dp in Week 4 to add
differential privacy noise on top.

Run (in 3 separate terminals):
    python federated/client.py --client-id 0 --server-address localhost:8080
    python federated/client.py --client-id 1 --server-address localhost:8080
    python federated/client.py --client-id 2 --server-address localhost:8080
"""

import argparse
import sys
import os
import numpy as np
import torch
import flwr as fl
from torch.utils.data import DataLoader

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model.unet3d import build_model, build_loss, build_metric
from data.synthetic_data import partition_dataset
from privacy.dp_utils import add_dp_noise


def get_parameters(model):
    return [val.cpu().numpy() for val in model.state_dict().values()]


def set_parameters(model, parameters):
    keys = list(model.state_dict().keys())
    new_state = {k: torch.tensor(v) for k, v in zip(keys, parameters)}
    model.load_state_dict(new_state, strict=True)


class HospitalClient(fl.client.NumPyClient):
    def __init__(self, client_id, n_clients=3, volume_size=48, local_epochs=1,
                 use_dp=False, dp_epsilon=5.0, non_iid=False, device=None):
        self.client_id = client_id
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = build_model().to(self.device)
        self.loss_fn = build_loss()
        self.metric_fn = build_metric()
        self.local_epochs = local_epochs
        self.use_dp = use_dp
        self.dp_epsilon = dp_epsilon

        # each node only ever sees ITS OWN partition — this is the "raw data
        # never leaves the hospital" guarantee in practice
        partitions = partition_dataset(n_total=n_clients * 20, n_clients=n_clients,
                                        volume_size=volume_size, iid=not non_iid)
        self.train_loader = DataLoader(partitions[client_id], batch_size=2, shuffle=True)

        print(f"[hospital-{client_id}] ready, {len(partitions[client_id])} local samples, device={self.device}")

    def get_parameters(self, config):
        return get_parameters(self.model)

    def fit(self, parameters, config):
        # 1. receive current global weights from the server
        received_params = [p.copy() for p in parameters]
        set_parameters(self.model, parameters)

        # 2. train LOCALLY on this hospital's private data only
        optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        self.model.train()
        for epoch in range(self.local_epochs):
            epoch_loss = 0.0
            for batch in self.train_loader:
                images = batch["image"].to(self.device)
                labels = batch["label"].to(self.device)
                optimizer.zero_grad()
                outputs = self.model(images)
                loss = self.loss_fn(outputs, labels)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            print(f"[hospital-{self.client_id}] local epoch {epoch+1}/{self.local_epochs} loss={epoch_loss/len(self.train_loader):.4f}")

        # 3. compute the UPDATE (delta), not the raw weights — this is what
        #    gets DP-noised and/or encrypted before leaving the hospital
        new_params = get_parameters(self.model)
        updates = [new - old for new, old in zip(new_params, received_params)]

        if self.use_dp:
            flat_shapes = [u.shape for u in updates]
            flat = np.concatenate([u.flatten() for u in updates])
            flat = add_dp_noise(flat, epsilon=self.dp_epsilon)
            idx = 0
            noised = []
            for shape in flat_shapes:
                n = int(np.prod(shape))
                noised.append(flat[idx:idx+n].reshape(shape))
                idx += n
            updates = noised
            print(f"[hospital-{self.client_id}] applied DP noise (epsilon={self.dp_epsilon})")

        # reconstruct weights-to-send = original + (possibly noised) update
        # (encryption, if enabled, happens at the Flower transport/strategy
        # layer — see federated/server.py + privacy/he_utils.py for the HE path)
        params_to_send = [old + u for old, u in zip(received_params, updates)]

        return params_to_send, len(self.train_loader.dataset), {}

    def evaluate(self, parameters, config):
        set_parameters(self.model, parameters)
        self.model.eval()
        self.metric_fn.reset()
        total_loss = 0.0
        with torch.no_grad():
            for batch in self.train_loader:
                images = batch["image"].to(self.device)
                labels = batch["label"].to(self.device)
                outputs = self.model(images)
                loss = self.loss_fn(outputs, labels)
                total_loss += loss.item()
                preds = (torch.sigmoid(outputs) > 0.5).float()
                self.metric_fn(y_pred=preds, y=labels)
        dice = self.metric_fn.aggregate().item()
        return total_loss / len(self.train_loader), len(self.train_loader.dataset), {"dice": dice}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--client-id", type=int, required=True)
    parser.add_argument("--n-clients", type=int, default=3)
    parser.add_argument("--server-address", type=str, default="localhost:8080")
    parser.add_argument("--volume-size", type=int, default=48)
    parser.add_argument("--local-epochs", type=int, default=1)
    parser.add_argument("--dp", action="store_true", help="enable differential privacy noise (Week 4)")
    parser.add_argument("--dp-epsilon", type=float, default=5.0)
    parser.add_argument("--non-iid", action="store_true", help="simulate heterogeneous hospital data distributions")
    args = parser.parse_args()

    client = HospitalClient(
        client_id=args.client_id,
        n_clients=args.n_clients,
        volume_size=args.volume_size,
        local_epochs=args.local_epochs,
        use_dp=args.dp,
        dp_epsilon=args.dp_epsilon,
        non_iid=args.non_iid,
    )
    fl.client.start_numpy_client(server_address=args.server_address, client=client)
