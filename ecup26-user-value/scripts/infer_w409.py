#!/usr/bin/env python3
"""T3.19 infer_w409: auto-discover best w409 models -> neural + ensemble + tabular hybrids.

Auto-discovers the best saved single-input model per kind (gru/lstm) for a given
window from models/neural/*.json (min best_rmsle). Optional --lr filter selects
only models trained with that lr (LB-confirmed optimum is 0.0005 — review
2026-08-29: fold-B prefers 0.0008 but LB prefers 0.0005, proxy->LB drift).

Produces (every output validated R10: user_id unique, len==250000):
  submit_neural_w{W}_{kind}.csv              single model, raw expm1
  submit_neural_w{W}_ensemble.csv            mean of log-space preds over kinds
  submit_hybrid_w{W}_{kind}_w{...}.csv       log-blend with tabular base (c=1.04 baked in)
  submit_hybrid_w{W}_ensemble_w{...}.csv

The hybrid recipe matches the confirmed winner (gru409v2_w0.3 = LB 1.650366):
  pred = expm1(w * log1p(neural) + (1 - w) * log1p(tabular))
The tabular base already carries the c=1.04 inflation — no extra dial here.

Usage:
  uv run python scripts/infer_w409.py --window 409 --lr 0.0005 \
      --ws 0.2,0.25,0.3,0.35,0.4
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from neural_models import SequenceModel, SequenceModelXLSTM  # noqa: E402

LOG1P_IDX = (2, 3, 4, 5, 6, 7, 8, 9, 10, 11)
TABULAR = (
    "submissions/submit_blend_actor_retuned_2026-08-20_161657_C1.A2-control_M1.A2-inflate-c04.csv"
)
MODELS_DIR = Path("models/neural")


def find_best_model(
    kind: str, window: int, models_dir: Path = MODELS_DIR, lr: float | None = None
) -> tuple[str, float] | None:
    """Return (stem, best_rmsle) of the best single-input saved model.

    Skips two-input models (json has ``static_dim``) — they need a second input.
    """
    best: tuple[str, float] | None = None
    for jp in sorted(models_dir.glob("*.json")):
        cfg = json.loads(jp.read_text())
        if cfg.get("model") != kind or cfg.get("window") != window:
            continue
        if "static_dim" in cfg:  # SequenceModelStatic -> needs static input
            continue
        if lr is not None and cfg.get("lr") != lr:
            continue
        r = cfg.get("best_rmsle")
        if r is None:
            continue
        if best is None or r < best[1]:
            best = (jp.stem, float(r))
    return best


def ensemble_logs(logs: list[np.ndarray]) -> np.ndarray:
    """Mean of log-space predictions over models."""
    return np.mean(np.stack(logs, axis=0), axis=0)


def hybrid_log(n_log: np.ndarray, t_log: np.ndarray, w: float) -> np.ndarray:
    """w * neural_log + (1-w) * tabular_log (winning recipe)."""
    return w * n_log + (1.0 - w) * t_log


def load_test(window: int) -> tuple[np.ndarray, np.ndarray]:
    z = np.load(f"data/sequences/seq_TEST_L{window}.npz")
    X = z["X"].astype(np.float32)
    X[:, :, LOG1P_IDX] = np.log1p(X[:, :, LOG1P_IDX])
    return X, z["user_ids"]


def load_model(name: str, dev: str) -> torch.nn.Module:
    cfg = json.loads((MODELS_DIR / f"{name}.json").read_text())
    if cfg.get("model") == "xlstm":
        m = SequenceModelXLSTM(d_in=16, hidden=cfg["hidden"], layers=cfg["layers"])
    else:
        m = SequenceModel(d_in=16, hidden=cfg["hidden"], layers=cfg["layers"], kind=cfg["model"])
    sd = torch.load(MODELS_DIR / f"{name}.pt", map_location="cpu")
    # models saved from a torch.compile'd module carry an "_orig_mod." prefix
    if any(k.startswith("_orig_mod.") for k in sd):
        sd = {k.replace("_orig_mod.", "", 1): v for k, v in sd.items()}
    m.load_state_dict(sd)
    return m.to(dev).eval()


def predict(m: torch.nn.Module, X: np.ndarray, dev: str) -> np.ndarray:
    ps = []
    with torch.no_grad():
        for i in range(0, len(X), 512):
            ps.append(m(torch.tensor(X[i : i + 512]).to(dev)).cpu().numpy())
    return np.concatenate(ps)


def save(out: pd.DataFrame, path: Path) -> None:
    assert out["user_id"].is_unique and len(out) == 250000, f"R10 failed: {path}"
    out.to_csv(path, index=False)
    print(f"  {path.name}: n={len(out)} mean={out['predict'].mean():.2f} unique=True")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--window", type=int, default=409)
    ap.add_argument("--kinds", type=str, default="gru,lstm")
    ap.add_argument(
        "--lr",
        type=float,
        default=0.0005,
        help="only consider models with this lr (LB-confirmed); omit for best fold-B",
    )
    ap.add_argument(
        "--lr-map",
        type=str,
        default="",
        help="per-kind lr overrides, e.g. 'gru:0.0005,xlstm:0.0003' (overrides --lr)",
    )
    ap.add_argument(
        "--tag",
        type=str,
        default="ensemble",
        help="name for the multi-kind log-mean output (hybrid filename tag)",
    )
    ap.add_argument("--ws", type=str, default="0.2,0.25,0.3,0.35,0.4")
    ap.add_argument("--tabular", type=str, default=TABULAR)
    args = ap.parse_args()

    kinds = [k.strip() for k in args.kinds.split(",") if k.strip()]
    ws = [float(x) for x in args.ws.split(",") if x.strip()]
    lr_map: dict[str, float] = {}
    if args.lr_map:
        for kv in args.lr_map.split(","):
            k, v = kv.split(":")
            lr_map[k.strip()] = float(v.strip())
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    X, uids = load_test(args.window)
    print(
        f"test seq: {X.shape} | device: {dev} | kinds={kinds} lr={args.lr} "
        f"lr_map={lr_map} ws={ws}",
        flush=True,
    )

    logs: dict[str, np.ndarray] = {}
    for kind in kinds:
        kind_lr = lr_map.get(kind, args.lr)
        hit = find_best_model(kind, args.window, lr=kind_lr)
        if hit is None:
            print(
                f"  !! no saved model for kind={kind} window={args.window} lr={kind_lr}",
                flush=True,
            )
            continue
        name, r = hit
        print(f"  {kind}: best saved = {name} (fold-B {r:.4f})", flush=True)
        m = load_model(name, dev)
        logs[kind] = predict(m, X, dev)
        save(
            pd.DataFrame({"user_id": uids, "predict": np.expm1(logs[kind])}),
            Path(f"submissions/submit_neural_w{args.window}_{kind}.csv"),
        )
    if not logs:
        print("  no models found — nothing to do", flush=True)
        return 1

    tab = pd.read_csv(args.tabular).set_index("user_id")
    tab_log = np.log1p(tab.reindex(pd.Index(uids))["predict"].fillna(0.0).to_numpy())

    tags: dict[str, np.ndarray] = dict(logs)
    if len(logs) > 1:
        tags[args.tag] = ensemble_logs(list(logs.values()))
        save(
            pd.DataFrame({"user_id": uids, "predict": np.expm1(tags[args.tag])}),
            Path(f"submissions/submit_neural_w{args.window}_{args.tag}.csv"),
        )

    for tag, n_log in tags.items():
        for w in ws:
            h = np.expm1(hybrid_log(n_log, tab_log, w))
            save(
                pd.DataFrame({"user_id": uids, "predict": h}),
                Path(f"submissions/submit_hybrid_w{args.window}_{tag}_w{w}.csv"),
            )
    return 0


if __name__ == "__main__":
    sys.exit(main())
