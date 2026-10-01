# T3 Experiment Protocol & Noise Criterion (2026-08-18)

> Why this doc exists: the leaderboard is compressed into ~0.0106 RMSLE
> (#1 Daniil Stepanov 1.645914 vs #78 Maxim Boguslavskiy 1.656517). Top-30 sit
> within ~0.005. The old "noise threshold" (±0.001 on a single fold) is therefore
> WRONG: a technique that moved fold B by -0.0007 was treated as "noise", but the
> whole distance from #78 to #1 is only ~15 such increments. **Every 0.0005 counts.**

## 1. Noise criterion (NEW — replaces single-fold point deltas)

Decisions are made by **sign consistency** across repeated shuffled folds with
replacement (paired bootstrap), never by a point delta on one fold.

Per seed k (default K=10):

1. bootstrap-resample the fold's training rows **with replacement** (same size, seeded);
2. train method A and B on the **same** resample;
3. score both on the fold's full infer+truth set;
4. record `delta_k = rmsle(A) - rmsle(B)` in log-space.

Report: `mean ± std` of delta, `P(A beats B) = fraction of seeds with delta<0`,
and a 95% bootstrap CI.

Decision rule:

| P(A beats B) | mean delta | Verdict |
|---|---|---|
| >= 0.90 | < 0 | **A WINS** |
| <= 0.10 | > 0 | **B WINS** |
| otherwise | any | **TIE** → re-run with K=20..40 |

Why not single-fold ±0.001: two techniques that were "KEPT" on fold B
(hybrid T3.8.10, log-calibration T3.8.6) both **DEGRADED on LB** (1.6672 vs
1.6588). Fold-B point deltas do not transfer. Sign-consistency across resamples
is the only signal with a chance to transfer to the LB.

Tool: `uv run python scripts/ab_validate.py --a <A> --b <B> --fold B --seeds 10`.

- **Mandatory fold chain: B + A + P + G** (T3.10.28). Fold G = extended gift
  season 2025 (cutoff 2025-02-14, truth Feb14-Mar15 = target window shifted -1y),
  cache `data/folds/giftG` (353k train rows, 3 origins).
- **Decision (rule 14, 2026-08-20)**: INTEGRATE only if WINS on ALL FOUR folds
  with P <= 0.1. BUT a sign-flip on ONE non-target fold (e.g. promo-fold P)
  is **NOT a reject** — it becomes a **CANDIDATE submission with an annotation**
  (`--task <ledger-ID>`, validated on LB). LB is the only ground truth (P6);
  the old "reject on P-flip" produced 5 false negatives (RCA P0). REJECT only
  on: loss on 2+ folds incl. target-fold G, or a stable negative sweep.
- **Regime probing (rule 15)**: LB experiments are "dial" designs — a parameter
  whose optimum maps to the data regime (shrink `c` ↔ zero_frac) is scanned;
  the winning value identifies the closest fold and sets fold weights.
- **Periodic review (rule 16)**: every N=5 submissions — pause, re-estimate the
  regime + fold weights + proxy→LB offset, reopen stale verdicts (ledger
  revisit_trigger), run an overfit check (LB moved but folds didn't?), and
  re-prioritize. Note in `docs/research/reviews/YYYY-MM-DD.md`.


## 2. Ticket template (each task = a SERIES of experiments)

Every ticket MUST fill in:

- **## Hypothesis** — what we believe and why (cite ledger / research / LB).
- **## Expectations** — concrete numbers and the condition under which they hold.
- **## Series of experiments** — E1, E2, … each with its own expectation.
- **## What to check** — the exact metric/output that proves or disproves.
- **## How to verify** — command + noise criterion + pass/fail gate.
- **## Sweeps (переборы)** — grid/ranges to search, in order.
- **## Compliance** — R-rules this ticket touches.
- **## Forbidden actions** — explicit do-nots with R-rule references.
- **## Result** — filled AFTER the run (proxy before → after + sign-consistency).

## 3. Metric conventions

- `rmsle(y_true, y_pred) = sqrt(mean((log1p(y_true) - log1p(max(0,y_pred)))^2))`.
- All A/B deltas are computed in **log-space** (pred_log vs log1p(truth)).
- LB ≈ proxy − 0.035 (empirical rule from note 003).

## 4. Known dead ends (do NOT re-propose — cite in tickets)

| Method | Result | Source |
|---|---|---|
| two-stage raw-space (p_buy × amount) | 2.02 REJECTED | note 003 line 80 |
| weighted single-stage clip(1+log1p(gmv),1,6) | 2.09 REJECTED | note 003 |
| LGBM tweedie objective | 2.53 vs 1.71 REJECTED | note 003 |
| ensemble lgb+xgb+cat | noise REJECTED | note 003 line 72 |
| features v3 / holiday / gift-season | noise REJECTED | note 003 |
| season-only (train Feb-Mar 2025) | 2.00 REJECTED | note 003 line 91 |
| per-bucket post-hoc calibration | LB 1.6672 DEGRADED | note 003 line 103 |
| hybrid OOF purchase_prob feature | LB 1.6672 DEGRADED | note 003 line 66 |
| neural ZILN / dense+MSE / transformer | 2.58 / 1.7262 REJECTED | note 003 |
| orders×AOV decomposition (E[orders]×E[AOV]) | mean 4.99 vs truth 66.9 REJECTED | T3.13.A1 (ledger 2026-08-23) |
| gift-origin injection + gift-weight (GMV) | mean 35.53 = control REJECTED | T3.13.A3 (ledger 2026-08-23) |
| macro GMV multiplier (отчёт +36%) | LB c>1.10 вредит (1.6575); fold-G c*=0.9 | M1.A1 + T3.13.A2 |

## 5. Key synthesis

The metric is **MSE in log-space**, so `log1p` + L2 is the Bayes-optimal loss
for RMSLE. Any **raw-space** zero-inflation loss (ZILN, tweedie, hurdle in raw
space) optimizes the WRONG thing and loses (2.53 / 2.58). Therefore every
zero-inflation fix must be applied **in log-space**: hard-zero threshold on
P(y>0), or the log-space hurdle `pred_log = P(y>0) * E[log1p(y)|y>0]`.

## 6. Empirical noise floor (measured 2026-08-18, fold B, M=500 paired test-bootstrap)

- std_delta of a single fold-B RMSLE estimate: **~0.00024** (baseline-vs-hurdle),
  ~0.00018 (baseline-vs-CatBoost), ~0.00014 (feature A/B).
- Practical rule: a single-fold delta < 0.0005 is UNRELIABLE; only sign-consistent
  deltas across folds/bootstrap transfer (hurdle +0.00043 on fold B was REJECTED
  because fold A showed -0.00178; CatBoost +0.00087 on fold B held with a fold-A TIE).

## 7. Confirmed lever (2026-08-18, T3.10.13)

**Log-space shrink of the production blend (c=0.95)** is the first validated
improvement that closes the #78->#1 gap:
- Zeros (46%) are over-predicted (pred_log ~1.42, bias -1.42) and carry ~50% of
the error^2; positives are near-calibrated. The additive log-space bias (T3.8.6)
was neutral (b~0) because the two cancel globally - only a MULTIPLICATIVE shrink
fixes the asymmetry.
- Bootstrap A/B M=500: fold B +0.00946 (P=0.00), fold A +0.00233 (P=0.00). No
sign-flip. Optimum c scales with the zero fraction (B=0.94, A=0.97; target ~0.484
wants c~0.94-0.95).
- This is ~10x the CatBoost gain and equals the #78 (1.6565) -> #1 (1.6459) gap.
Production: experiment_blend_production.py --shrink 0.95.
