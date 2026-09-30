"""One-command CLI demo: python scripts/run_demo.py  (writes docs/results/*)"""
import json, sys, tempfile
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from backend.config import Settings, ROOT
from backend.database.results import ResultsStore
from backend.federated.server import FederatedServer
from backend.utils.data_gen import generate_all

ROUNDS, EPOCHS, LR = 10, 2, 0.1
out = ROOT / "docs" / "results"; out.mkdir(parents=True, exist_ok=True)
tmp = Path(tempfile.mkdtemp())
generate_all(tmp / "nodes")
srv = FederatedServer(Settings(tmp / "nodes", tmp / "r.db"), ResultsStore(tmp / "r.db"))
for i in (1, 2, 3):
    srv.register(f"hospital_{i}", f"Hospital {'ABC'[i-1]}")
srv.start(seed=0)
for _ in range(ROUNDS):
    srv.run_round(EPOCHS, LR)
cmp_ = srv.run_baseline(ROUNDS * EPOCHS, LR)
g, h = srv.store.global_rounds(), srv.store.hospital_rounds()
json.dump(dict(config=dict(rounds=ROUNDS, epochs=EPOCHS, lr=LR), global_rounds=g,
               hospital_rounds=h, comparison=cmp_), open(out / "results.json", "w"), indent=1)

print("Round | Hospital   | LocalAcc | GlobalAcc | LocalLoss | GlobalLoss")
for r in h:
    print(f"{r['round']:>5} | {r['hospital_id']:<10} | {r['local_accuracy']:.3f}    | {r['global_accuracy']:.3f}     | {r['local_loss']:.3f}     | {r['global_loss']:.3f}")
print("\nPooled test-set comparison (all hospitals' held-out test data):")
print(f"{'model':<24}{'acc':>7}{'prec':>7}{'rec':>7}{'f1':>7}")
for m in cmp_["local_only"] + [cmp_["federated"]]:
    print(f"{m['model']:<24}{m['accuracy']:>7.3f}{m['precision']:>7.3f}{m['recall']:>7.3f}{m['f1']:>7.3f}")
print("Federated confusion matrix [[TN,FP],[FN,TP]]:", cmp_["federated"]["confusion_matrix"])

x = [r["round"] for r in g]
fig, ax = plt.subplots(1, 3, figsize=(14, 3.6))
for k in ("accuracy", "precision", "recall", "f1"):
    ax[0].plot(x, [r[k] for r in g], marker="o", label=k)
ax[0].set(title="Global model vs round", xlabel="round"); ax[0].legend()
ax[1].plot(x, [r["loss"] for r in g], marker="o", color="tab:red"); ax[1].set(title="Global loss vs round", xlabel="round")
for hid in sorted({r["hospital_id"] for r in h}):
    ax[2].plot(x, [r["global_accuracy"] for r in h if r["hospital_id"] == hid], marker="o", label=f"{hid} (global model)")
ax[2].set(title="Hospital-wise accuracy of global model", xlabel="round"); ax[2].legend(fontsize=7)
plt.tight_layout(); plt.savefig(out / "training_curves.png", dpi=120)
