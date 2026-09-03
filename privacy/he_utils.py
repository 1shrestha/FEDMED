"""
he_utils.py  (WEEK 3 DELIVERABLE)
----------------------------------
Homomorphic encryption helpers using TenSEAL (CKKS scheme).

IMPORTANT CONCEPTUAL NOTE (read before you build on this):
CKKS only supports ADDITION and SCALAR MULTIPLICATION on ciphertext — not
arbitrary neural-net ops. That is exactly enough for FedAvg aggregation
(summing weight vectors), but NOT enough to do anything else on encrypted
data. So the pattern is:

    1. Each client flattens its model weights into a single vector.
    2. Each client encrypts that vector with a SHARED public context
       (all hospitals use the same encryption parameters/keys so their
       ciphertexts are compatible for aggregation).
    3. Clients send ciphertext to the server.
    4. The server homomorphically SUMS the ciphertexts (it can do this
       WITHOUT ever decrypting — it never sees raw weights).
    5. The encrypted sum is sent back to a party holding the secret key
       (in a real deployment this is usually a separate "key holder" that
       is NOT the aggregation server, to preserve the privacy guarantee —
       see note at bottom) which decrypts (passing the secret context to
       decrypt_and_average) and divides by n_clients to get the plaintext
       averaged weights.

For a course/prototype project, it's acceptable (and common) to have the
central server also hold the secret key for decryption at step 5, AS LONG
AS you clearly document that this means the server could technically
decrypt individual client updates if it deviated from the aggregate-then-
decrypt protocol. True zero-trust setups split the secret key across
parties with Secure Multiparty Computation (SMPC) - flag this as a
"future work" discussion point in your Final Review, it shows you
understand the limitation rather than glossing over it.
"""

import numpy as np
import tenseal as ts
import torch


def create_context(poly_modulus_degree=8192, scale_bits=40):
    """
    Creates the shared CKKS encryption context. All hospital clients AND the
    server must use the SAME context (or at least the same public key) or
    their ciphertexts will be mathematically incompatible for aggregation.

    In this prototype, the server generates the context (with the secret
    key) and distributes only the PUBLIC part to clients via
    `make_public_context_bytes` below. Clients encrypt with the public
    context; only the server can decrypt.
    """
    context = ts.context(
        ts.SCHEME_TYPE.CKKS,
        poly_modulus_degree=poly_modulus_degree,
        coeff_mod_bit_sizes=[60, scale_bits, scale_bits, 60],
    )
    context.generate_galois_keys()
    context.global_scale = 2 ** scale_bits
    return context


def make_public_context_bytes(context: ts.Context) -> bytes:
    """Serializes a copy of the context WITHOUT the secret key, safe to send to clients."""
    public_ctx = context.copy()
    public_ctx.make_context_public()
    return public_ctx.serialize()


def flatten_state_dict(state_dict):
    """Model weights (a dict of tensors) -> a single flat numpy vector + shape metadata to unflatten later."""
    shapes = {k: v.shape for k, v in state_dict.items()}
    flat = np.concatenate([v.detach().cpu().numpy().flatten() for v in state_dict.values()])
    return flat, shapes


def unflatten_to_state_dict(flat: np.ndarray, shapes: dict, reference_state_dict):
    """Inverse of flatten_state_dict — rebuilds a torch state_dict from a flat vector."""
    new_state = {}
    idx = 0
    for k, shape in shapes.items():
        n = int(np.prod(shape))
        chunk = flat[idx: idx + n].reshape(shape)
        new_state[k] = torch.tensor(chunk, dtype=reference_state_dict[k].dtype)
        idx += n
    return new_state


CHUNK_SIZE = 4096  # CKKS vectors have a max slot count tied to poly_modulus_degree; chunk large weight vectors


def encrypt_weights(flat_weights: np.ndarray, context: ts.Context):
    """
    Encrypts a (potentially huge) flat weight vector as a LIST of CKKS
    ciphertext chunks (real 3D U-Nets have millions of params — one CKKS
    vector can't hold them all at once).
    """
    chunks = [flat_weights[i:i + CHUNK_SIZE] for i in range(0, len(flat_weights), CHUNK_SIZE)]
    return [ts.ckks_vector(context, chunk) for chunk in chunks]


def sum_encrypted_weights(list_of_encrypted_weight_lists):
    """
    Homomorphically sums N clients' encrypted weight chunk-lists.
    THE SERVER RUNS THIS. It never sees plaintext at any point here.
    """
    n_clients = len(list_of_encrypted_weight_lists)
    n_chunks = len(list_of_encrypted_weight_lists[0])
    summed = []
    for chunk_idx in range(n_chunks):
        acc = list_of_encrypted_weight_lists[0][chunk_idx]
        for client_idx in range(1, n_clients):
            acc = acc + list_of_encrypted_weight_lists[client_idx][chunk_idx]
        summed.append(acc)
    return summed


def decrypt_and_average(summed_encrypted_chunks, n_clients, total_len, secret_context: ts.Context = None):
    """
    Only the party holding the SECRET key can call this — pass the full
    (non-public) context here, e.g. the original `context` returned by
    `create_context()`. Ciphertext chunks built on a client's *public*
    context carry no secret key themselves, so it must be supplied
    explicitly at decrypt time.
    Returns the plaintext averaged flat vector.
    """
    secret_key = secret_context.secret_key() if secret_context is not None else None
    parts = [chunk.decrypt(secret_key=secret_key) if secret_key is not None else chunk.decrypt()
             for chunk in summed_encrypted_chunks]
    flat = np.concatenate(parts)[:total_len]
    return flat / n_clients
