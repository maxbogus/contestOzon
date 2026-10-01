#!/usr/bin/env python3
"""T3.19 inference: load saved neural models -> test predictions -> submission CSVs."""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
from neural_models import SequenceModel  # noqa: E402

LOG1P_IDX = (2, 3, 4, 5, 6, 7, 8, 9, 10, 11)
MODELS = {
    "gru": "gru_h96_l2_lr0.0005_w240_r1.6799",
    "lstm": "lstm_h128_l2_lr0.0003_w240_r1.6823",
    "transformer": "transformer_h128_l2_lr0.0003_w240_r1.6846",
}
TABULAR = "submissions/submit_blend_actor_retuned_2026-08-20_161657_C1.A2-control_M1.A2-inflate-c04.csv"


def load_test(window=240):
    z = np.load(Path("data/sequences/seq_TEST_L240.npz"))
    X = z["X"][:, -window:, :].astype(np.float32)
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
        for i in range(0, len(X), 1024):
            ps.append(m(torch.tensor(X[i:i+1024]).to(dev)).cpu().numpy())
    return np.concatenate(ps)


def main() -> int:
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    X, uids = load_test(240)
    print("test seq:", X.shape, "device:", dev)
    logs = {}
    for tag, name in MODELS.items():
        m = load_model(name, dev)
        lp = predict(m, X, dev)
        logs[tag] = lp
        pred = np.expm1(lp)
        out = pd.DataFrame({"user_id": uids, "predict": pred})
        assert out["user_id"].is_unique and len(out) == 250000
        out.to_csv("submissions/submit_neural_" + tag + ".csv", index=False)
        print("  ", tag, "mean:", round(float(pred.mean()), 2))
    ens_log = np.mean([logs[t] for t in logs], axis=0)
    ens = np.expm1(ens_log)
    out = pd.DataFrame({"user_id": uids, "predict": ens})
    out.to_csv("submissions/submit_neural_ensemble.csv", index=False)
    print("  ensemble mean:", round(float(ens.mean()), 2))
    tab = pd.read_csv(TABULAR)
    tab_log = np.log1p(tab.set_index("user_id").reindex(uids)["predict"].fillna(0.0).to_numpy())
    for w in (0.7, 0.5):
        hlog = w * ens_log + (1 - w) * tab_log
        hy = np.expm1(hlog)
        out = pd.DataFrame({"user_id": uids, "predict": hy})
        out.to_csv("submissions/submit_hybrid_w" + str(w) + ".csv", index=False)
        print("  hybrid w", w, "mean:", round(float(hy.mean()), 2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
