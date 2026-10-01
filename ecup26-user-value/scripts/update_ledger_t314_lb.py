#!/usr/bin/env python3
"""Update decision ledger with T3.14 LB verdicts (2026-08-23 evening)."""
from __future__ import annotations

import json
from pathlib import Path

LEDGER = Path("docs/research/decision_ledger_v2.jsonl")

entries: list[dict] = []
for line in LEDGER.read_text().strip().splitlines():
    entries.append(json.loads(line))

# 1) T3.14.A2 -> FALSIFIED on LB (gift-origin injection costs +0.00013)
for e in entries:
    if e["id"] == "T3.14.A2":
        e["status"] = "FALSIFIED"
        e["observed"] += (
            "; LB 5 slots: noextra_c1.045=1.651892 (wash), giftnorm_c1.045=1.651966"
            " (+0.00013), gw2_c1.045=1.652373 (+0.00054) -> gift-origin/weight"
            " FALSIFIED на LB; главный сигнал - нормализация, но она тоже wash на LB"
        )

# 2) T3.14.A3: c* dial CONFIRMED at 1.045
entries.append({
    "id": "T3.14.A3",
    "type": "atomic_check",
    "parents": ["T3.14", "c-dial"],
    "status": "CONFIRMED",
    "date": "2026-08-23",
    "hypothesis": "Оптимум shrink c для calv3 остаётся 1.045 (как для calv2); 1.02 и 1.06 хуже",
    "artifact_before": "submissions/*G3.2-noextra_c1.0[246]*.csv + calv2_c1045 (LB 1.651833)",
    "command": "LB 3 слота на noextra: c=1.02/1.045/1.06",
    "metric": "LB RMSLE",
    "gate_pre": "c=1.045 лучше обоих соседей",
    "observed": "c1.02=1.652444, c1.045=1.651892, c1.06=1.652916 -> c*=1.045 подтверждён",
    "verdict_conditions": "c*=1.045 железно; дальнейшие dial-точки не нужны",
    "revisit_trigger": "нет",
    "source": ["submissions/INDEX.md"],
})

# 3) T3.14.A4: normalization = WASH on LB (proxy->LB drift, P6)
entries.append({
    "id": "T3.14.A4",
    "type": "atomic_check",
    "parents": ["T3.14", "WS-1", "P6"],
    "status": "CONFIRMED",
    "date": "2026-08-23",
    "hypothesis": "История-нормализация (per-day rates) улучшит LB: локально fold-G -0.019, fold-B -0.0023",
    "artifact_before": "build_features.py +6 норм-фич (env ECUP26_HIST_NORM_FEATURES), actor3_calv3 folds",
    "command": "production calv3 noextra c=1.045 (LB)",
    "metric": "LB RMSLE vs calv2_c1045 (1.651833)",
    "gate_pre": "LB < 1.651833",
    "observed": "noextra_c1.045 = 1.651892 (+0.00006, wash). fold-G -0.019 НЕ перенёсся на LB -> proxy->LB дрейф (P6). LB определяется ранжированием, не агрегатным сюрджем",
    "verdict_conditions": "нормализация = CANDIDATE-на-private (ставка: private 80% gift-подобный), не улучшение public",
    "revisit_trigger": "private-режим gift-подобный -> пересмотреть в финале (2 выбранных решения)",
    "source": ["submissions/INDEX.md", "docs/research/reviews/2026-08-23.md"],
})

LEDGER.write_text("\n".join(json.dumps(e, ensure_ascii=False) for e in entries) + "\n")
print(f"ledger updated: {len(entries)} entries")
