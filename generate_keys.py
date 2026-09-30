"""
Run ONCE per deployment (or per key-rotation period), by the party trusted
to hold the secret key -- in production this should be an isolated
key-management service, not the same container that runs FedAvg logic.

Produces:
  keys/secret_context.bin   -> stays on the server, never copied to a node
  keys/public_context.bin   -> distributed to every client node
"""
import os
from encryption import he_utils

OUT_DIR = os.environ.get("KEYS_DIR", "/keys")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    ctx = he_utils.create_secret_context()

    secret_path = os.path.join(OUT_DIR, "secret_context.bin")
    public_path = os.path.join(OUT_DIR, "public_context.bin")

    with open(secret_path, "wb") as f:
        f.write(he_utils.secret_context_bytes(ctx))
    with open(public_path, "wb") as f:
        f.write(he_utils.public_context_bytes(ctx))

    print(f"Secret context written to {secret_path} (keep on server only)")
    print(f"Public context written to {public_path} (copy to every client node)")


if __name__ == "__main__":
    main()
