#!/usr/bin/env python3
"""T3.19 infer w360/w409 saved models -> neural CSVs + tabular hybrids."""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
from neural_models import SequenceModel  # noqa: E402

LOG1P_IDX = (2, 3, 4, 5, 6, 7, 8, 9, 10, 11)
TABULAR = "submissions/submit_blend_actor_retuned_2026-08-20_161657_C1.A2-control_M1.A2-inflate-c04.csv"
MODELS = [
    ("gru360", "gru_h128_l2_lr0.0003_w360_r1.6769", 360),
    ("lstm360", "lstm_h128_l2_lr0.0003_w360_r1.6761", 360),
    ("gru409", "gru_h128_l2_lr0.0003_w409_r1.6745", 409),
    ("lstm409", "lstm_h128_l2_lr0.0003_w409_r1.6765", 409),
]


def load_test(window):
    z = np.load(f"data/sequences/seq_TEST_L{window}.npz")
    X = z["X"].astype(np.float32)
    X[:, :, LOG1P_IDX] = np.log1p(X[:, :, LOG1P_IDX])
    return X, z["user_ids"]


def load_model(name, dev):
    cfg = json.loads(open("models/neural/" + name + ".json").read())
    m = SequenceModel(d_in=16, hidden=cfg["hidden"], layers=cfg["layers"], kind=cfg["model"])
    m.load_state_dict(torch.load("models/neural/" + name + ".pt", map_location="cpu"))
    return m.to(dev).eval()


def predict(m, X, dev):
    ps = []
    with torch.no_grad():
        for i in range(0, len(X), 512):
            ps.append(m(torch.tensor(X[i:i+512]).to(dev)).cpu().numpy())
    return np.concatenate(ps)


def main() -> int:
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tab = pd.read_csv(TABULAR).set_index("user_id")
    logs = {}
    uids = None
    for tag, name, window in MODELS:
        X, uids = load_test(window)
        print(f"{tag}: test X {X.shape}, dev={dev}", flush=True)
        m = load_model(name, dev)
        lp = predict(m, X, dev)
        logs[tag] = lp
        pred = np.expm1(lp)
        out = pd.DataFrame({"user_id": uids, "predict": pred})
        assert out["user_id"].is_unique and len(out) == 250000
        out.to_csv(f"submissions/submit_neural_{tag}.csv", index=False)
        print(f"  {tag} mean={float(pred.mean()):.2f} -> submit_neural_{tag}.csv", flush=True)
    tab_log = np.log1p(tab.reindex(pd.Index(uids))["predict"].fillna(0.0).to_numpy())
    for tag in logs:
        for w in (0.08, 0.1, 0.12, 0.15):
            h = np.expm1(w * logs[tag] + (1 - w) * tab_log)
            out = pd.DataFrame({"user_id": uids, "predict": h})
            assert out["user_id"].is_unique and len(out) == 250000
            out.to_csv(f"submissions/submit_hybrid_{tag}_w{w}.csv", index=False)
            print(f"  hybrid_{tag}_w{w} mean={float(h.mean()):.2f}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
