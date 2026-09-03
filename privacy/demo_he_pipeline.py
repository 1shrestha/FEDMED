"""
demo_he_pipeline.py  (run this for your Week 3 review demo)
--------------------------------------------------------------
Proves, standalone, that the encrypt -> homomorphically-sum -> decrypt
pipeline recovers the correct FedAvg average without the server ever
seeing plaintext weights. Good evidence to show in your Federated Audit.

Run:
    python privacy/demo_he_pipeline.py
"""

import numpy as np
import tenseal as ts

from he_utils import (
    create_context, make_public_context_bytes,
    encrypt_weights, sum_encrypted_weights, decrypt_and_average,
)


def main():
    print("[1] Server generates the CKKS context (holds the secret key)")
    server_context = create_context()

    print("[2] Server distributes a PUBLIC (no secret key) copy to hospitals")
    public_bytes = make_public_context_bytes(server_context)
    client_context = ts.context_from(public_bytes)
    print(f"    public context is {len(public_bytes)} bytes")

    n_clients = 3
    vector_len = 20000  # stand-in for a flattened (small) model
    client_weights = [np.random.uniform(-1, 1, vector_len) for _ in range(n_clients)]

    print(f"[3] Each of {n_clients} hospitals encrypts its local weights (server never sees these)")
    encrypted_per_client = [encrypt_weights(w, client_context) for w in client_weights]

    print("[4] Server homomorphically sums ciphertexts — still never decrypts")
    summed = sum_encrypted_weights(encrypted_per_client)

    print("[5] Only the secret-key holder decrypts + averages")
    result = decrypt_and_average(summed, n_clients=n_clients, total_len=vector_len, secret_context=server_context)

    expected = np.mean(client_weights, axis=0)
    max_err = np.max(np.abs(result - expected))
    print(f"\nmax abs error vs plaintext FedAvg: {max_err:.2e}")
    print("PASS — encrypted aggregation matches plaintext average" if max_err < 1e-4 else "CHECK YOUR PARAMS")


if __name__ == "__main__":
    main()
