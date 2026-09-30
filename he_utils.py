"""
Homomorphic encryption layer (CKKS via TenSEAL) for FedMed.

Design:
- A single CKKS context (with secret key) is generated ONCE by a trusted
  key-holder (the central server, in a real deployment ideally an external
  key-management/aggregator service that is NOT the same operator as any
  hospital) and distributed to every client node as a *public* context
  (secret key stripped) so nodes can encrypt but never decrypt.
- Client nodes encrypt their local weight-delta vectors and send only
  ciphertexts to the server.
- The server homomorphically sums ciphertexts across nodes (CKKS supports
  addition on encrypted data) and returns the aggregated ciphertext.
- Only the key-holder (holding the secret context) can decrypt the summed
  result, then divides by n to get the FedAvg mean before updating the
  global model.

This means raw weights are never visible to the server in plaintext, and no
single hospital ever sees another hospital's update.
"""
from __future__ import annotations
import pickle
from typing import List

import numpy as np
import tenseal as ts

# ---- Context management -----------------------------------------------

def create_secret_context(poly_modulus_degree: int = 8192,
                           coeff_mod_bit_sizes=(60, 40, 40, 60),
                           global_scale: float = 2 ** 40) -> ts.Context:
    ctx = ts.context(
        ts.SCHEME_TYPE.CKKS,
        poly_modulus_degree=poly_modulus_degree,
        coeff_mod_bit_sizes=list(coeff_mod_bit_sizes),
    )
    ctx.generate_galois_keys()
    ctx.global_scale = global_scale
    return ctx


def public_context_bytes(secret_ctx: ts.Context) -> bytes:
    """Serialize a copy of the context with the secret key stripped, for
    distribution to client nodes."""
    public_ctx = secret_ctx.copy()
    public_ctx.make_context_public()
    return public_ctx.serialize()


def secret_context_bytes(secret_ctx: ts.Context) -> bytes:
    return secret_ctx.serialize(save_secret_key=True)


def load_context(ctx_bytes: bytes) -> ts.Context:
    return ts.context_from(ctx_bytes)


# ---- Weight <-> flat vector helpers ------------------------------------

def flatten_weights(weights: List[np.ndarray]) -> tuple[np.ndarray, list]:
    shapes = [w.shape for w in weights]
    flat = np.concatenate([w.ravel() for w in weights])
    return flat, shapes


def unflatten_weights(flat: np.ndarray, shapes: list) -> List[np.ndarray]:
    out, idx = [], 0
    for shape in shapes:
        size = int(np.prod(shape))
        out.append(flat[idx: idx + size].reshape(shape))
        idx += size
    return out


# ---- Encrypt / aggregate / decrypt -------------------------------------

CHUNK = 4096  # CKKS vector encoding batch size


def encrypt_weights(public_ctx: ts.Context, weights: List[np.ndarray]) -> bytes:
    flat, shapes = flatten_weights(weights)
    chunks = [ts.ckks_vector(public_ctx, flat[i:i + CHUNK].tolist())
              for i in range(0, len(flat), CHUNK)]
    payload = {
        "shapes": shapes,
        "n_chunks": len(chunks),
        "chunk_size": CHUNK,
        "total_len": len(flat),
        "chunks": [c.serialize() for c in chunks],
    }
    return pickle.dumps(payload)


def sum_encrypted(ctx: ts.Context, encrypted_payloads: List[bytes]) -> bytes:
    """Homomorphically sum N clients' encrypted weight vectors. Runs on the
    server, which never sees plaintext."""
    payloads = [pickle.loads(p) for p in encrypted_payloads]
    n_chunks = payloads[0]["n_chunks"]
    shapes = payloads[0]["shapes"]
    total_len = payloads[0]["total_len"]

    summed_chunks = []
    for i in range(n_chunks):
        acc = ts.ckks_vector_from(ctx, payloads[0]["chunks"][i])
        for p in payloads[1:]:
            acc += ts.ckks_vector_from(ctx, p["chunks"][i])
        summed_chunks.append(acc.serialize())

    return pickle.dumps({
        "shapes": shapes, "n_chunks": n_chunks,
        "chunk_size": CHUNK, "total_len": total_len,
        "chunks": summed_chunks,
    })


def decrypt_and_average(secret_ctx: ts.Context, summed_payload: bytes,
                         n_clients: int) -> List[np.ndarray]:
    payload = pickle.loads(summed_payload)
    flat = np.zeros(payload["total_len"], dtype=np.float64)
    for i, chunk_bytes in enumerate(payload["chunks"]):
        vec = ts.ckks_vector_from(secret_ctx, chunk_bytes)
        dec = np.array(vec.decrypt())
        start = i * payload["chunk_size"]
        flat[start:start + len(dec)] = dec
    flat /= n_clients
    return unflatten_weights(flat, payload["shapes"])
