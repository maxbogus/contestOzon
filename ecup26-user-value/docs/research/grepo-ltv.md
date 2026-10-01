# GRePO-LTV — research notes

> **Status:** closed (T3.6.2)
> **Author:** Maxim Boguslavskiy
> **Created:** 2026-08-16
> **Source:** Chen et al., *Mini-Game Lifetime Value Prediction in WeChat* (GRePO-LTV),
> arXiv:2506.11037v3 [cs.LG], KDD ADS Track 2025; v1 21 May 2025, v3 13 Aug 2025.
> **DOI:** https://doi.org/10.48550/arxiv.2506.11037
> **Compliance:** R1 (open-source; notes only, no code copying), R5 (cite, do not copy).

## 1. TL;DR

Two problems in LTV prediction when purchase rate is extremely low (~0.1%) and horizon
tasks are correlated:

1. **Data scarcity.** Few purchases → weak supervisory signal for a single LTV model.
   Paper uses **graph representation learning** over (user, item, attribute) to share
   signal across users with similar structural roles.
2. **Multi-horizon interference.** Predicting 7-day LTV is heavily entangled with 3-day
   LTV; bad 3d predictions snowball into bad 7d. Paper uses **Pareto-optimization**
   across horizons so no single horizon dominates.

## 2. Method outline

- **Graph representation:** heterogeneous graph of users, items, item attributes; node
  embeddings trained via message passing on (user–item purchase events) and
  (item–attribute relations). The structured prior compensates for sparse purchases.
- **Pareto-Optimal head:** training step enumerates 3d / 7d / 30d tasks, computes
  per-sample gradients, projects them onto each other so that no task gradient is
  dominated. Equivalent to MGDA (Multiple-Gradient Descent Algorithm).
- **Combined loss:** convex combination of Pareto-optimal weights and per-task LTV loss.

## 3. Applicability to E-CUP T3 (user-value, RMSLE)

Our task: predict **sum GMV over 30 days** for 250k users from daily aggregate history
(no item_id — only user, day, search/cat flags, counts, GMV). Metric RMSLE.

### 3.1 Domain caveat (must not be glossed over)

GRePO-LTV is deployed for **WeChat mini-games**, not e-commerce. The structural
assumptions (item catalog relatively stable, user-item purchase graph dense enough to
benefit from message passing) are different. Anything we adopt needs to be re-validated
on our data, not assumed.

### 3.2 Applicability matrix

| Component | Applicable? | Notes |
|---|---|---|
| **Zero-inflation / sparse supervision** | ✅ Yes | Our purchase rate is similarly low — most users have 0 GMV on most days. We already ship `models/ziln.py` (T3.6.7) for this. GRePO is independent validation that this is the right family of techniques. |
| **Multi-horizon prediction (3d / 7d / 30d)** | ✅ Yes | Direct transfer. We can train 3 auxiliary heads at 7d / 14d / 30d horizons and use Pareto-balanced loss. Cheap to add; gives ablation a "temporal" feature group. |
| **Pareto-MGDA gradient balancing** | ⚠️ Conditional | Reasonable for multi-horizon auxiliary tasks; not needed if we only have one RMSLE objective. Add only if we adopt multi-horizon heads. |
| **Heterogeneous graph (user–item–attribute)** | ❌ No | We have **no item_id** in the data, only user×day aggregates. Cannot build the graph. Document the rejection reason. |
| **Game-specific context features** (level, session, payment tier) | ❌ No | Not in our data. |

## 4. Concrete things to take into T3 backlog

1. **Multi-horizon auxiliary heads** (T3.6.12 ablation, "temporal" group). Train at
   7d / 14d / 30d horizons; use Pareto-balanced loss as a drop-in variant of the main
   RMSLE head. Strongest concrete win from this paper.
2. **Reaffirm ZILN** (T3.6.7 already shipped). GRePO gives independent motivation for
   modeling zero-inflation explicitly, not via plain log1p.
3. **Do NOT build a graph.** Explicit rejection. Document once, link from here when
   the question comes up again.

## 5. What we explicitly reject

- Graph representation learning — no item_id in data.
- Game-specific features — wrong domain.
- Full Pareto-MGDA on the single 30d RMSLE objective — Pareto is meaningful only when
  multiple objectives fight.

## 6. References

- arXiv:2506.11037 — https://arxiv.org/abs/2506.11037
- Authors: Aochuan Chen, Yifan Niu, Ziqi Gao, Yujie Sun, Shoujun Liu, Gong Chen,
  Yang Liu, Jia Li
- Venue: KDD ADS Track 2025
- Related: Vietri et al., "Pareto Optimal Loss Surfaces" (MGDA foundation)
