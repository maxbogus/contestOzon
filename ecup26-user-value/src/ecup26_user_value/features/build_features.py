"""Feature engineering for T3 LTV forecasting (T3.7.3).

Per .clinerules/02-ecup-rules.md:
- R1: open-source (pandas, numpy)
- R6: reproducible (deterministic, no random)
- R8: NOT imported by run.py until T3.7.6 (isolated module)

Per meta-prompt T3.7.5 Cycle #1. Four feature groups, all computed
**only from rows before ref_date** (no future leakage):

1. Calendar: day_of_week, month, quarter, is_weekend, is_holiday_ru
2. Rolling windows: gmv/orders avg over last 7/14/30 days
3. Cross-features: recency * frequency, etc.
4. Tenure: days_since_first_purchase, is_new_customer

Optimization note (2026-08-17): the original per-user Python loop was
O(N_users x N_rows) (~7.6e12 comparisons at 250k users x 30.6M rows =>
hours). All features are now computed with vectorized pandas groupby
aggregations (~7s on the full train).

API:
    build_features(df, ref_date, ...) -> pd.DataFrame

Returns a per-user feature DataFrame (one row per user_id).
"""

from __future__ import annotations

import logging
import os

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


#: Russian public holidays + extended non-working days 2025-2026 (date objects).
#: Format: (year, month, day). Includes the official holidays AND the transfer/
#: bridge days (extended weekend blocks per the government production calendar):
#:   2025: Jan 1-8 (Jan 7 = Christmas), Feb 22-23, Mar 8-9, May 1-4, May 9-11,
#:         Jun 12-14, Nov 3-4.
#:   2026: Jan 1-8 (Jan 7 = Christmas), Feb 21-23, Mar 7-9, May 1-3, May 9-11,
#:         Jun 12-14, Nov 4.
_RU_HOLIDAYS: set[tuple[int, int, int]] = {
    # 2025
    (2025, 1, 1), (2025, 1, 2), (2025, 1, 3), (2025, 1, 4), (2025, 1, 5),
    (2025, 1, 6), (2025, 1, 7), (2025, 1, 8),
    (2025, 2, 22), (2025, 2, 23),
    (2025, 3, 8), (2025, 3, 9),
    (2025, 5, 1), (2025, 5, 2), (2025, 5, 4),
    (2025, 5, 9), (2025, 5, 10), (2025, 5, 11),
    (2025, 6, 12), (2025, 6, 13), (2025, 6, 14),
    (2025, 11, 3), (2025, 11, 4),
    # 2026
    (2026, 1, 1), (2026, 1, 2), (2026, 1, 3), (2026, 1, 4), (2026, 1, 5),
    (2026, 1, 6), (2026, 1, 7), (2026, 1, 8),
    (2026, 2, 21), (2026, 2, 22), (2026, 2, 23),
    (2026, 3, 7), (2026, 3, 8), (2026, 3, 9),
    (2026, 5, 1), (2026, 5, 2), (2026, 5, 3),
    (2026, 5, 9), (2026, 5, 10), (2026, 5, 11),
    (2026, 6, 12), (2026, 6, 13), (2026, 6, 14),
    (2026, 11, 4),
}

#: Commercial/gift days (NOT official holidays) - the E-CUP gift window (T3.14/Г1.5).
#: 2025 days (14.02 Valentine, 23.02 Defender, 08.03 Women) are in the past for
#: all walk-forward origins >= 2025-06-18 -> leak-free via `past` clipping; they
#: feed the per-user day-level GMV features. The 2026 days feed only the
#: prediction-window scalar (days_to_next_gift_day): at infer ref=2026-02-13 the
#: next gift day is 2026-02-14 (1 day) - the same "gift-regime" marker that the
#: 2025 gift origins carry, so the model can transfer the season boost.
_GIFT_DAYS_2025: tuple[tuple[int, int, int], ...] = (
    (2025, 2, 14),
    (2025, 2, 23),
    (2025, 3, 8),
)
_GIFT_DAYS: tuple[tuple[int, int, int], ...] = _GIFT_DAYS_2025 + (
    (2026, 2, 14),
    (2026, 2, 23),
    (2026, 3, 8),
)

#: Separator value for "no holiday in the future" (days_to_next_*).
_NO_FUTURE_HOLIDAY = 1000.0


def _days_until_next(dates: set[tuple[int, int, int]], ref_date: pd.Timestamp) -> float:
    """Days from ref_date to the NEXT date in `dates` (strictly in the future)."""
    fut = [pd.Timestamp(*d) for d in dates if pd.Timestamp(*d) > ref_date]
    if not fut:
        return _NO_FUTURE_HOLIDAY
    return float(min((f - ref_date).days for f in fut))


def _count_in_next_30d(dates: set[tuple[int, int, int]], ref_date: pd.Timestamp) -> float:
    """Number of dates in (ref_date, ref_date+30] (the prediction horizon)."""
    end = ref_date + pd.Timedelta(days=30)
    return float(
        sum(1 for d in dates if ref_date < pd.Timestamp(*d) <= end)
    )


def _is_holiday(d: pd.Timestamp) -> int:
    """Return 1 if the date is a Russian public holiday, else 0."""
    return 1 if (d.year, d.month, d.day) in _RU_HOLIDAYS else 0


def _days_to_nearest_holiday(d: pd.Timestamp) -> float:
    """Min absolute day distance from a date to any RU holiday (v3 feature)."""
    best = 1000.0
    for year, month, day in _RU_HOLIDAYS:
        h = pd.Timestamp(year, month, day)
        best = min(best, abs(float((h - d).days)))
    return best


#: Default rolling windows (days) for the gmv/orders average features.
#: T3.9.2: +120/180d KEPT - consistent -0.0007 on both folds (CV 1.7150 vs 1.7155).
DEFAULT_ROLLING_WINDOWS: tuple[int, ...] = (7, 14, 30, 60, 90, 120, 180)


def _feature_columns(rolling_windows: tuple[int, ...]) -> list[str]:
    """Canonical feature column order (matches the pre-optimization schema)."""
    cols = ["user_id", "day_of_week", "month", "quarter", "is_weekend", "is_holiday_ru"]
    for w in rolling_windows:
        cols += [f"gmv_avg_{w}d", f"orders_avg_{w}d"]
    cols += [
        "gmv_max_30d",
        "n_active_days_30d",
        "frequency",
        "recency",
        "T",
        "monetary_value",
        "recency_x_frequency",
        "recency_x_avg_order_value",
        "frequency_x_monetary",
        "days_since_first_purchase",
        "is_new_customer",
        # T3.14/WS-1: history-length-invariant rates (per-day normalization).
        "orders_per_active_day",
        "gmv_per_active_day",
        "activity_rate",
        "gmv_per_day_alive",
        "orders_per_day_alive",
        "recency_rate",
    ]
    return cols


#: Full daily-schema columns (task3_user_value_data.md) used for rich features.
_ACTIVITY_COUNT_COLS: tuple[str, ...] = (
    "searches",
    "to_cart",
    "to_ord",
    "search_to_cart",
    "search_to_ord",
    "cat_to_cart",
    "cat_to_ord",
    "gmv_search",
    "gmv_cat",
    "gmv",
)
_WINDOW_ACTIVITY_COLS: tuple[str, ...] = (
    "gmv_search",
    "gmv_cat",
    "searches",
    "to_cart",
    "to_ord",
    "search_to_ord",
    "cat_to_ord",
    "gmv",
)
_RECENCY_ACTIVITY_COLS: tuple[str, ...] = (
    "gmv",
    "to_ord",
    "to_cart",
    "searches",
    "gmv_search",
    "gmv_cat",
    "search_to_ord",
    "cat_to_ord",
)

#: Recency value for users with no activity in a channel (never-active bucket).
_NEVER_ACTIVE_RECENCY = 999.0


def _activity_features(
    past: pd.DataFrame,
    uid: str,
    d: str,
    ref_date: pd.Timestamp,
    windows: tuple[int, ...],
) -> dict[str, pd.Series]:
    """Per-user activity features from the full daily schema (when present).

    All features are computed only from rows <= ref_date (no future leakage):
    - lifetime totals per channel (gmv_search, gmv_cat, orders, carts, searches)
    - conversion rates (orders/searches, carts/searches) and channel share
    - windowed per-day averages for key activity counters
    - recency per channel (days since last order/cart/search/GMV activity)
    """
    feats: dict[str, pd.Series] = {}
    cols = set(past.columns)
    grp = past.groupby(uid)

    for col in _ACTIVITY_COUNT_COLS:
        if col in cols:
            feats[f"{col}_total"] = grp[col].sum().astype(float)

    if {"searches", "to_ord"} <= cols:
        feats["conv_to_ord"] = feats["to_ord_total"] / feats["searches_total"].clip(lower=1.0)
    if {"searches", "to_cart"} <= cols:
        feats["conv_to_cart"] = feats["to_cart_total"] / feats["searches_total"].clip(lower=1.0)
    if {"gmv_search", "gmv"} <= cols:
        feats["gmv_search_share"] = feats["gmv_search_total"] / feats["gmv_total"].clip(lower=1e-9)
    if {"search", "cat"} <= cols:
        n_days = past.groupby(uid)[d].nunique().astype(float)
        feats["search_day_share"] = grp["search"].sum().astype(float) / n_days.clip(lower=1.0)
        feats["cat_day_share"] = grp["cat"].sum().astype(float) / n_days.clip(lower=1.0)

    for w in windows:
        cutoff = ref_date - pd.Timedelta(days=w)
        win = past[past[d] > cutoff]
        wg = win.groupby(uid)
        for col in _WINDOW_ACTIVITY_COLS:
            if col in cols:
                feats[f"{col}_{w}d"] = wg[col].sum().astype(float) / w

    for col in _RECENCY_ACTIVITY_COLS:
        if col in cols:
            active = past[past[col] > 0]
            if active.empty:
                continue
            last = active.groupby(uid)[d].max()
            feats[f"recency_{col}"] = (ref_date - last).dt.days.astype(float)

    # v3 (T3.8.5): windowed conversions (orders/carts per search) + log1p totals
    for w in windows:
        cutoff = ref_date - pd.Timedelta(days=w)
        win = past[past[d] > cutoff]
        wg = win.groupby(uid)
        if "searches" in cols:
            denom = wg["searches"].sum().astype(float).clip(lower=1.0)
            if "to_ord" in cols:
                feats[f"conv_to_ord_{w}d"] = wg["to_ord"].sum().astype(float) / denom
            if "to_cart" in cols:
                feats[f"conv_to_cart_{w}d"] = wg["to_cart"].sum().astype(float) / denom
    for name in ("gmv_total", "to_ord_total", "searches_total"):
        if name in feats:
            feats[f"log1p_{name}"] = np.log1p(feats[name].clip(lower=0.0))
    return feats


def _absence_features(
    past: pd.DataFrame,
    uid: str,
    d: str,
    ref_date: pd.Timestamp,
) -> dict[str, pd.Series]:
    """T3.10.4: absence / gap structure of the THINNED series (env-gated).

    The data is sparse rows (only days WITH activity; full grid would be ~102M
    rows for 250k x 409d). These features capture the structure of inactivity:
    - gap_mean / gap_max / gap_std: distribution of days between active days
    - n_long_gaps: count of inactivity gaps longer than 14 days (lifetime)
    - gap_last: days since the last active day (the trailing gap to ref_date)

    Enabled via ECUP26_ABSENCE_FEATURES=1 (dev-only; OFF in production by
    default so the submission schema stays unchanged until the A/B passes).
    """
    feats: dict[str, pd.Series] = {}
    srt = past.sort_values([uid, d])
    gaps = srt.groupby(uid)[d].diff().dt.days
    g = pd.DataFrame({"uid": srt[uid].values, "gap": gaps.astype(np.float32).values})
    g = g.dropna()
    if g.empty:
        return feats
    gg = g.groupby("uid")["gap"]
    feats["gap_mean"] = gg.mean().astype(float)
    feats["gap_max"] = gg.max().astype(float)
    feats["gap_std"] = gg.std().astype(float)
    feats["n_long_gaps"] = (g["gap"] > 14).groupby(g["uid"]).sum().astype(float)
    last_act = past.groupby(uid)[d].max()
    feats["gap_last"] = (ref_date - last_act).dt.days.astype(float)
    return feats


def _behavioral_features(
    past: pd.DataFrame,
    uid: str,
    d: str,
    price_col: str,
) -> dict[str, pd.Series]:
    """T3.10.12: behavioral purchase-TIMING traits (env-gated).

    The target window (Feb 14 - Mar 15 2026) is a gift season with Feb 23 and
    Mar 8. People buy in ADVANCE (up to ~2 weeks before), at the LAST MINUTE,
    or LATE (after the holiday). A user's timing pattern is a stable trait.
    All windows are 2025 facts -> leak-free for the 2026 target.

    Non-overlapping partition of the 2025 gift season (Feb 14 - Mar 15):
      adv_feb23 = [Feb 14, Feb 20]   (early, ahead of Feb 23)
      lm_feb23  = [Feb 21, Feb 23]   (last-minute for Feb 23)
      adv_mar8  = [Feb 24, Mar 05]   (between holidays = advance for Mar 8)
      lm_mar8   = [Mar 06, Mar 08]   (last-minute for Mar 8)
      late      = [Mar 09, Mar 15]   (after Mar 8, still in the 30d window)
    Shares (planner/lastmin/late) sum to 1 - the SHAPE is the new behavioral
    signal; totals alone would duplicate feb_mar_gmv_2025.
    Mass-buying baseline: pre-Sep-1 (Aug 15-31) and pre-NY (Dec 15-31).

    Enabled via ECUP26_BEHAVIOR_FEATURES=1 (dev-only A/B, OFF in production).
    """
    feats: dict[str, pd.Series] = {}
    windows: dict[str, tuple[str, str]] = {
        "adv_feb23": ("2025-02-14", "2025-02-20"),
        "lm_feb23": ("2025-02-21", "2025-02-23"),
        "adv_mar8": ("2025-02-24", "2025-03-05"),
        "lm_mar8": ("2025-03-06", "2025-03-08"),
        "late": ("2025-03-09", "2025-03-15"),
    }
    raw: dict[str, pd.Series] = {}
    for name, (a, b) in windows.items():
        win = past[(past[d] >= pd.Timestamp(a)) & (past[d] <= pd.Timestamp(b))]
        g = win.groupby(uid)[price_col].sum()
        raw[name] = g
        feats[f"gmv_{name}_2025"] = np.log1p(g).astype(float)

    g_adv = raw["adv_feb23"].add(raw["adv_mar8"], fill_value=0.0)
    g_lm = raw["lm_feb23"].add(raw["lm_mar8"], fill_value=0.0)
    g_late = raw["late"]
    total = g_adv.add(g_lm, fill_value=0.0).add(g_late, fill_value=0.0).clip(lower=0.0)
    denom = total.clip(lower=1.0)
    feats["flag_adv_buyer_2025"] = (g_adv > 0).astype(float)
    feats["flag_lm_buyer_2025"] = (g_lm > 0).astype(float)
    feats["flag_late_buyer_2025"] = (g_late > 0).astype(float)
    feats["planner_share_2025"] = (g_adv / denom).astype(float)
    feats["lastmin_share_2025"] = (g_lm / denom).astype(float)
    feats["late_share_2025"] = (g_late / denom).astype(float)

    for name, (a, b) in (("sep1", ("2025-08-15", "2025-08-31")),
                         ("ny", ("2025-12-15", "2025-12-31"))):
        win = past[(past[d] >= pd.Timestamp(a)) & (past[d] <= pd.Timestamp(b))]
        g = win.groupby(uid)[price_col].sum()
        feats[f"gmv_pre_{name}_2025"] = np.log1p(g).astype(float)
    mass = (np.expm1(feats["gmv_pre_sep1_2025"].fillna(0.0))
            + np.expm1(feats["gmv_pre_ny_2025"].fillna(0.0)))
    feats["gift_vs_mass_2025"] = (total / (mass + 1.0)).astype(float)
    return feats

def _actor_features(
    past: pd.DataFrame,
    uid: str,
    d: str,
    price_col: str,
) -> dict[str, pd.Series]:
    """T3.10.17: actor-type behavioral features (env-gated).

    Captures buyer actor types from the 18 columns (leak-free, rows <= ref_date):
    - IMPULSIVE: instant_buy_share = to_ord/(to_ord+to_cart) (buys w/o cart)
    - PLANNED:   cart_backlog = to_cart - to_ord (unresolved cart "fills up")
    - EVENT-TIED (salary): buy_gap_mean + buy_gap_cv (regular vs random intervals)
    - RANDOM/cat: gmv_cat_share (browses catalog, buys there)
    - WHALE:     gmv_cv_daily + max_day_gmv_share (concentration of spend)
    - dom_mode: dominant day-of-month of purchases (salary-day buyer 1/15/25)

    Enabled via ECUP26_ACTOR_FEATURES=1 (dev-only A/B, OFF in production).
    """
    feats: dict[str, pd.Series] = {}
    cols = set(past.columns)

    if {"to_ord", "to_cart"} <= cols:
        to_ord_t = past.groupby(uid)["to_ord"].sum().astype(float)
        to_cart_t = past.groupby(uid)["to_cart"].sum().astype(float)
        denom = (to_ord_t + to_cart_t).clip(lower=1.0)
        feats["instant_buy_share"] = (to_ord_t / denom).astype(float)
        feats["cart_backlog"] = (to_cart_t - to_ord_t).clip(lower=0.0).astype(float)
        feats["max_day_cart"] = past.groupby(uid)["to_cart"].max().astype(float)
        feats["max_day_ord"] = past.groupby(uid)["to_ord"].max().astype(float)

    if {"to_ord"} <= cols:
        ord_days = past[past["to_ord"] > 0]
        if not ord_days.empty:
            srt = ord_days.sort_values([uid, d])
            gaps = srt.groupby(uid)[d].diff().dt.days
            g = pd.DataFrame({"uid": srt[uid].values, "gap": gaps.astype(np.float32).values})
            g = g.dropna()
            if not g.empty:
                gg = g.groupby("uid")["gap"]
                m = gg.mean()
                feats["buy_gap_mean"] = m.astype(float)
                feats["buy_gap_cv"] = (gg.std() / m.clip(lower=1e-9)).astype(float)
                feats["dom_mode"] = (
                    ord_days[d].dt.day.groupby(ord_days[uid].values)
                    .agg(lambda x: int(pd.Series(x).mode().iloc[0]))
                    .astype(float))

    if {"gmv_cat", "gmv"} <= cols:
        gmv = past.groupby(uid)["gmv"].sum().clip(lower=1e-9)
        gmv_cat = past.groupby(uid)["gmv_cat"].sum()
        feats["gmv_cat_share"] = (gmv_cat / gmv).astype(float)

    daily = past.groupby([uid, d])[price_col].sum()
    if not daily.empty:
        user = daily.index.get_level_values(0)
        mean_d = daily.groupby(user).mean()
        std_d = daily.groupby(user).std()
        max_d = daily.groupby(user).max()
        total_d = daily.groupby(user).sum().clip(lower=1e-9)
        feats["gmv_cv_daily"] = (std_d / mean_d.clip(lower=1e-9)).astype(float)
        feats["max_day_gmv_share"] = (max_d / total_d).astype(float)

    # --- T3.10.26 funnel chains + cart->order lag (separate gate) ---
    if os.environ.get("ECUP26_FUNNEL_FEATURES", "0") == "1":
        # --- T3.10.17+: funnel chains + cart->order lag (env-gated with the actors) ---
        if {"search_to_cart", "cat_to_cart", "to_cart", "to_ord", "searches"} <= cols:
            s_c = past.groupby(uid)["search_to_cart"].sum().astype(float)
            c_c = past.groupby(uid)["cat_to_cart"].sum().astype(float)
            t_o = past.groupby(uid)["to_ord"].sum().astype(float)
            t_c = past.groupby(uid)["to_cart"].sum().astype(float)
            den = (s_c + c_c).clip(lower=1.0)
            feats["search_to_cart_ratio"] = (s_c / den).astype(float)
            feats["impulse_coef"] = (c_c / den).astype(float)
            feats["abandon_rate"] = ((t_c - t_o).clip(lower=0.0) / t_c.clip(lower=1.0)).astype(float)
            sear = past.groupby(uid)["searches"].sum().clip(lower=1.0)
            feats["search_to_ord_rate"] = (past.groupby(uid)["search_to_ord"].sum().astype(float) / sear).astype(float)
            _h_stc = (past["search_to_cart"] > 0).astype(int)
            feats["search_intent"] = (_h_stc.groupby(past[uid].values).sum().astype(float) / sear).astype(float)
            s_ord = past.groupby(uid)["search_to_ord"].sum().clip(lower=1.0)
            feats["avg_search_gmv"] = (past.groupby(uid)["gmv_search"].sum().astype(float) / s_ord).astype(float)
        if {"cat_to_cart", "cat"} <= cols:
            cat_days = past[past["cat"] > 0].groupby(uid)[d].nunique().astype(float)
            c2c_days = past[past["cat_to_cart"] > 0].groupby(uid)[d].nunique().astype(float)
            feats["cat_to_cart_rate"] = (c2c_days / cat_days.clip(lower=1.0)).astype(float)

        if {"to_cart", "to_ord"} <= cols:
            # memory-lean lag: process only cart/order rows via merge_asof, not the
            # full 30M-row sort+copy (that OOM'd the 30GB dev box on the fold rebuild).
            carts = past[past["to_cart"] > 0][[uid, d]].copy()
            carts["_cart_date"] = carts[d]
            carts = carts.sort_values(d)
            orders = past[past["to_ord"] > 0][[uid, d]].sort_values(d)
            if not carts.empty and not orders.empty:
                merged = pd.merge_asof(orders, carts, on=d, by=uid, direction="backward")
                lag = (merged[d] - merged["_cart_date"]).dt.days
                if lag.notna().any():
                    lu = merged[uid]
                    feats["cart_to_order_lag_mean"] = lag.groupby(lu.values).mean().astype(float)
                    feats["cart_to_order_lag_max"] = lag.groupby(lu.values).max().astype(float)
    return feats

#: Russian/Ozon seasonal windows (2025, fully in the past for all folds).
_SEASONAL_WINDOWS: dict[str, tuple[str, str]] = {
    "ny": ("2025-12-15", "2026-01-15"),
    "gift": ("2025-02-14", "2025-03-08"),
    "spring": ("2025-04-01", "2025-05-31"),
    "summer": ("2025-06-01", "2025-08-31"),
    "promo_1111": ("2025-10-23", "2025-11-11"),
    "promo_bf": ("2025-11-14", "2025-12-03"),
}


def _seasonal_features(
    past: pd.DataFrame,
    uid: str,
    d: str,
    ref_date: pd.Timestamp,
    price_col: str,
) -> dict[str, pd.Series]:
    """Per-user Russian/Ozon seasonal, promo and gift features (T3.8.7).

    All windows are bounded by ref_date (no future leakage). Captures
    holiday-season spending shares, gift-buyer behaviour (the prediction window
    Feb14-Mar15 IS the gift season) and post-New-Year recovery.
    """
    feats: dict[str, pd.Series] = {}
    total = past.groupby(uid)[price_col].sum().clip(lower=1e-9)
    for name, (a, b) in _SEASONAL_WINDOWS.items():
        # always emit the column so the schema is uniform across walk-forward
        # origins and train/infer (empty windows -> NaN -> fillna 0 later).
        win = past[(past[d] >= pd.Timestamp(a)) & (past[d] <= pd.Timestamp(b))]
        ws = win.groupby(uid)[price_col].sum()
        feats[f"gmv_share_{name}"] = (ws / total).astype(float)

    gift = past[(past[d] >= pd.Timestamp("2025-02-23")) & (past[d] <= pd.Timestamp("2025-03-08"))]
    g = gift.groupby(uid)[price_col].sum()
    feats["gift_buyer_2025"] = (g > 0).astype(float)
    feats["feb_mar_gmv_2025"] = np.log1p(g).astype(float)

    dec = past[(past[d] >= pd.Timestamp("2025-12-01")) & (past[d] <= pd.Timestamp("2025-12-31"))]
    nov = past[(past[d] >= pd.Timestamp("2025-11-01")) & (past[d] <= pd.Timestamp("2025-11-30"))]
    ds = dec.groupby(uid)[price_col].sum()
    ns = nov.groupby(uid)[price_col].sum()
    feats["dec_nov_ratio"] = (ds / ns.clip(lower=1.0)).astype(float)

    ny_end = pd.Timestamp("2026-01-11")
    last_act = past.groupby(uid)[d].max()
    feats["days_since_ny_activity"] = (last_act - ny_end).dt.days.clip(lower=-1).astype(float)
    return feats


#: T3.10.15: gift-holiday purchase-window flags (2025 gift season). Leak-free:
#: windows are clipped to rows <= ref_date via `past` inside build_features.
_GIFT_HOLIDAYS_2025: tuple[tuple[pd.Timestamp, str], ...] = (
    (pd.Timestamp("2025-02-23"), "feb23"),
    (pd.Timestamp("2025-03-08"), "mar8"),
)


def _holiday_prepost_features(
    past: pd.DataFrame,
    uid: str,
    d: str,
    price_col: str,
) -> dict[str, pd.Series]:
    """T3.10.15: pre/post-gift-holiday purchase flags (max 6 = 2 holidays x 3 windows).

    People buy IN ADVANCE (up to ~2 weeks before Feb 23 / Mar 8, confirmed by the
    fold-G analysis) and with delay. Each flag = 1 if the user purchased in the
    window around the holiday (pre-7d / pre-2d / post-7d). Computed only from
    `past` (rows <= ref_date) so there is no future leakage.
    """
    feats: dict[str, pd.Series] = {}
    for holiday, tag in _GIFT_HOLIDAYS_2025:
        for pre_days, name in ((7, "is_pre_holiday_7d"), (2, "is_pre_holiday_2d")):
            win = past[
                (past[d] >= holiday - pd.Timedelta(days=pre_days)) & (past[d] < holiday)
            ]
            feats[f"{name}_{tag}"] = (win.groupby(uid)[price_col].sum() > 0).astype(float)
        win = past[(past[d] > holiday) & (past[d] <= holiday + pd.Timedelta(days=7))]
        feats[f"is_post_holiday_7d_{tag}"] = (win.groupby(uid)[price_col].sum() > 0).astype(float)
    return feats


def _holiday_day_features(
    past: pd.DataFrame,
    uid: str,
    d: str,
    price_col: str,
    ref_date: pd.Timestamp,
    index: pd.Index,
) -> dict[str, pd.Series]:
    """T3.14/Г1.5: precise holiday-day + prediction-window-structure features.

    Two groups:
    1. Per-user, day-level: log1p GMV on each exact 2025 gift day (14.02/23.02/08.03)
       + count of active gift days. Computed only from `past` (<= ref_date) so the
       gift origins (2025-02-13) see 0 - no leakage.
    2. Per-origin scalars (broadcast to all users): days from ref_date to the NEXT
       RU holiday / gift day and the number of RU holidays inside the 30d horizon.
       These VARY across walk-forward origins, so the model can learn the
       "gift-regime" structure (few days to a holiday + 2 holidays in window =>
       boost active buyers) and apply it at inference (ref = 2026-02-13, where
       days_to_next_gift_day = 1 because of 14.02).
    """
    feats: dict[str, pd.Series] = {}
    for y, m, day in _GIFT_DAYS_2025:
        ts = pd.Timestamp(y, m, day)
        g = past[past[d] == ts].groupby(uid)[price_col].sum()
        feats[f"gmv_giftday_{y}_{m:02d}"] = np.log1p(g).astype(float)
    gift_dates = [pd.Timestamp(*gd) for gd in _GIFT_DAYS_2025]
    n_days = past[past[d].isin(gift_dates)].groupby(uid)[d].nunique()
    feats["n_gift_holiday_days_2025"] = n_days.astype(float)
    # Window-structure scalars (same value for every user at a given origin).
    feats["days_to_next_ru_holiday"] = pd.Series(
        _days_until_next(_RU_HOLIDAYS, ref_date), index=index, dtype=float
    )
    feats["n_ru_holidays_30d"] = pd.Series(
        _count_in_next_30d(_RU_HOLIDAYS, ref_date), index=index, dtype=float
    )
    feats["days_to_next_gift_day"] = pd.Series(
        _days_until_next(set(_GIFT_DAYS), ref_date), index=index, dtype=float
    )
    return feats


def _reactivation_features(
    past: pd.DataFrame,
    uid: str,
    d: str,
    price_col: str,
    ref_date: pd.Timestamp,
) -> dict[str, pd.Series]:
    """T3.16: reactivation / activity-decay features (dormant -> active).

    The test window (Feb-Mar 2026) is a post-Q4 decline; the users who still
    spend are the ones RAMPING BACK UP (reactivated by ads). These features
    capture who is "waking up" relative to their own cadence:
      - reactivated_last_{7,14,30}d: 1 if the most recent order came after a
        >30-day purchase gap and is inside the window before ref_date.
      - recency_ratio: days since last order / (mean order interval + 1); > 2
        means the user is overdue relative to their cadence (reactivation-prone).
      - activity_acceleration: (orders_7d - orders_prev_7d) / (orders_prev_7d + 1)
        (o14/o7 zero-filled first so a dormant-then-active user gets a positive
        jump and an active-then-dormant one a negative value, not NaN).
    Leak-free: everything is computed from `past` (rows <= ref_date).
    """
    feats: dict[str, pd.Series] = {}
    ord_col = "to_ord" if "to_ord" in past.columns else None
    active = past if ord_col is None else past[past[ord_col] > 0]
    if active.empty:
        return feats
    srt = active.sort_values([uid, d]).copy()
    # gap (days) from the PREVIOUS order to each row; last() per user = the gap
    # before the most recent order (NaN for single-order users). Index = group key.
    srt["_gap"] = srt.groupby(uid)[d].diff().dt.days
    grp = srt.groupby(uid)
    last = grp[d].last()
    gap = grp["_gap"].last().astype(float)
    days_since_last = (ref_date - last).dt.days.astype(float)
    for w in (7, 14, 30):
        react = (gap > 30.0).fillna(False) & (days_since_last <= w)
        feats[f"reactivated_last_{w}d"] = react.astype(float)
    first = grp[d].first()
    n_orders = grp.size().astype(float)
    # mean interval between orders over the user's ACTIVE span (not the dormant
    # tail): (last - first) / (n-1). A user overdue vs their own cadence gets
    # days_since_last / mean_interval > 2 (the reactivation-prone flag).
    active_span = (last - first).dt.days.clip(lower=1.0)
    mean_interval = active_span / (n_orders - 1).clip(lower=1.0)
    feats["recency_ratio"] = (days_since_last / (mean_interval + 1.0)).clip(upper=20.0)
    if ord_col is not None:
        w7 = past[past[d] > ref_date - pd.Timedelta(days=7)]
        w14 = past[
            (past[d] <= ref_date - pd.Timedelta(days=7))
            & (past[d] > ref_date - pd.Timedelta(days=14))
        ]
        all_u = past[uid].unique()
        o7 = w7.groupby(uid)[ord_col].sum().astype(float).reindex(all_u).fillna(0.0)
        o14 = w14.groupby(uid)[ord_col].sum().astype(float).reindex(all_u).fillna(0.0)
        feats["activity_acceleration"] = (o7 - o14) / (o14 + 1.0)
    # NOTE: yoy_* (vs same calendar window one year earlier) was removed: it is
    # only computable for ref >= 2026-01-31 (needs the 2024 same-window, but the
    # data starts 2025-01-01), so it can NEVER appear in a walk-forward training
    # origin -> 0 training signal + a train/infer schema mismatch (T3.16 hotfix).
    return feats


def _markov_state_features(
    past: pd.DataFrame,
    uid: str,
    d: str,
    ref_date: pd.Timestamp,
) -> dict[str, pd.Series]:
    """T3.17: per-user Markov state-transition features (T3.10.18 revival).

    States per active day: 0=browsing (search/cat, no cart/order), 1=carting
    (to_cart>0, no order), 2=buying (to_ord>0). A gap >= 7 days between active
    days is a DORMANT period that ends in a WAKE-UP (the T3.10.18 signal:
    Dormant->Active rate 1.99x in the gift season).

    Features (all per-user, history-normalized, leak-free from `past`):
      current_state       ordinal 0/1/2 (3=dormant if no active day in last 7d)
      days_in_state       days since the last active day
      p_buy_active        buying days / active days (base purchase propensity)
      p_cart_to_buy       Cart->Buy / Cart-as-source (per-user conversion)
      p_browse_to_buy     Browse->Buy / Browse-as-source
      p_buy_repeat        Buy->Buy / Buy-as-source (stickiness)
      n_wakeups_total     count of dormant->active transitions
      p_wakeup            wake-ups per day alive (reactivation-prone trait)
    """
    feats: dict[str, pd.Series] = {}
    has_cart = "to_cart" in past.columns
    has_ord = "to_ord" in past.columns
    work = past.copy()
    if has_cart and has_ord:
        work["_state"] = np.where(
            work["to_ord"] > 0, 2,
            np.where(work["to_cart"] > 0, 1, 0),
        ).astype(np.int8)
    else:
        work["_state"] = np.int8(0)
    srt = work.sort_values([uid, d]).copy()
    g = srt.groupby(uid, sort=False)
    srt["_prev_state"] = g["_state"].shift(1)
    srt["_gap"] = g[d].diff().dt.days

    all_u = past[uid].unique()
    first = g[d].first()
    last = g[d].last()
    days_alive = (ref_date - first).dt.days.clip(lower=1.0)
    days_since = (ref_date - last).dt.days.astype(float)

    # active-day state counts (per user)
    st_counts = work.groupby(uid)["_state"].value_counts().unstack(fill_value=0)
    n_active = work.groupby(uid).size().astype(float)
    p_buy_active = (
        st_counts.get(2, pd.Series(0.0, index=n_active.index)).astype(float) / n_active
    ).reindex(all_u).fillna(0.0)

    # transitions between consecutive active days with gap <= 6 (continuous chain)
    chain = srt.dropna(subset=["_gap"])
    chain = chain[chain["_gap"] <= 6]
    wake = srt.dropna(subset=["_gap"])
    wake = wake[wake["_gap"] >= 7]

    def _cnt(mask_prev, mask_state):
        return (
            chain[mask_prev & mask_state].groupby(uid).size().astype(float)
        )
    n_cart_src = _cnt(chain["_prev_state"] == 1, chain["_state"] >= 0)
    n_cart2buy = _cnt(chain["_prev_state"] == 1, chain["_state"] == 2)
    n_browse_src = _cnt(chain["_prev_state"] == 0, chain["_state"] >= 0)
    n_browse2buy = _cnt(chain["_prev_state"] == 0, chain["_state"] == 2)
    n_buy_src = _cnt(chain["_prev_state"] == 2, chain["_state"] >= 0)
    n_buy2buy = _cnt(chain["_prev_state"] == 2, chain["_state"] == 2)
    n_wakeups = wake.groupby(uid).size().astype(float)

    feats["p_cart_to_buy"] = (n_cart2buy / n_cart_src.clip(lower=1.0)).reindex(all_u).fillna(0.0)
    feats["p_browse_to_buy"] = (n_browse2buy / n_browse_src.clip(lower=1.0)).reindex(all_u).fillna(0.0)
    feats["p_buy_repeat"] = (n_buy2buy / n_buy_src.clip(lower=1.0)).reindex(all_u).fillna(0.0)
    feats["p_buy_active"] = p_buy_active
    feats["n_wakeups_total"] = n_wakeups.reindex(all_u).fillna(0.0)
    feats["p_wakeup"] = (n_wakeups / days_alive).reindex(all_u).fillna(0.0)

    # current state + days in it
    last_state = g["_state"].last()
    cur = np.where((days_since >= 7).reindex(all_u).fillna(True).to_numpy(), 3,
                   last_state.reindex(all_u).fillna(0).to_numpy())
    feats["current_state"] = pd.Series(cur, index=all_u, dtype=float)
    feats["days_in_state"] = days_since.reindex(all_u).fillna(409.0)  # dormant = big
    return feats


#: Full 2025/2026 Ozon sale calendar (public promos; dates approximate, may shift).
#: Covers sales MISSING from _SEASONAL_WINDOWS: Фантазируй, Изумляй, Зимняя одежда,
#: Радуй!, Удивляй! (2026), Финальная, Корзина удачи, ДР Ozon, O!Sale.
_SALES_2025: tuple[tuple[str, str], ...] = (
    ("2025-01-02", "2025-01-10"),  # Корзина удачи
    ("2025-01-08", "2025-01-28"),  # Зимняя распродажа одежды (бюджет 500M)
    ("2025-01-30", "2025-02-19"),  # Фантазируй
    ("2025-02-10", "2025-03-08"),  # Fitness SHOCK (партнёрская)
    ("2025-02-20", "2025-03-12"),  # Изумляй (подарки к 23.02/8.03)
    ("2025-02-21", "2025-03-07"),  # AWM (партнёрская)
    ("2025-04-08", "2025-04-22"),  # Всё в дом! Ремонт
    ("2025-05-22", "2025-06-25"),  # O!Sale
    ("2025-06-01", "2025-07-02"),  # День рождения Ozon (до -85%)
    ("2025-09-01", "2025-09-15"),  # Back to School (приблизительно)
    ("2025-10-23", "2025-12-03"),  # 11.11 + Чёрная пятница
    ("2025-12-04", "2025-12-29"),  # Финальная распродажа года (1B)
)
_SALES_2026: tuple[tuple[str, str], ...] = (
    ("2026-01-08", "2026-01-28"),  # Зимняя распродажа одежды 2026 (500M)
    ("2026-01-29", "2026-02-25"),  # Радуй! (1B) - частично видим до катоффа 13.02
)
#: TV-ad-campaign periods (public data, budgets 500M-1B).
_AD_PERIODS: tuple[tuple[str, str], ...] = (
    ("2025-01-08", "2025-01-28"),
    ("2025-01-30", "2025-03-12"),
    ("2025-12-04", "2025-12-29"),
    ("2026-01-29", "2026-02-25"),
)
_DORMANT_DAYS = 30  #: order gap (days) that counts as a "wake-up"


def _calendar_v2_features(
    past: pd.DataFrame,
    uid: str,
    d: str,
    ref_date: pd.Timestamp,
    price_col: str,
) -> dict[str, pd.Series]:
    """T3.11: calendar-v2 features (sale/ad day flags + wake-up + freq-accel).

    Leak-free: everything is computed from `past` (rows <= ref_date), so 2026
    sale windows after the cutoff are empty (zeros). Captures the 2026
    gift-dense regime (Valentine -> Defender -> Women's + Радуй!/Удивляй!):
    users who buy inside sales, dormant users who wake up during sales, and
    frequency acceleration (Q1-2026 macro direction, public/unverified).

    Enabled via ECUP26_CALENDAR_V2_FEATURES=1 (dev-only A/B, OFF in production).
    Sub-ablation: ECUP26_CALV2_WAKE=1 / ECUP26_CALV2_FREQ=1 / ECUP26_CALV2_SALE=1
    enable only the wake-up / freq-accel / sale-day groups (for ablation, T3.11).
    """
    feats: dict[str, pd.Series] = {}
    dates = pd.to_datetime(past[d])
    all_on = os.environ.get("ECUP26_CALENDAR_V2_FEATURES", "0") == "1"
    wake_on = all_on or os.environ.get("ECUP26_CALV2_WAKE", "0") == "1"
    freq_on = all_on or os.environ.get("ECUP26_CALV2_FREQ", "0") == "1"
    sale_on = all_on or os.environ.get("ECUP26_CALV2_SALE", "0") == "1"

    # --- T3.11.C: day flags -> user aggregates (sale/ad) -----------------------
    if sale_on:
        in_sale = pd.Series(False, index=past.index)
        for a, b in _SALES_2025 + _SALES_2026:
            in_sale |= (dates >= pd.Timestamp(a)) & (dates <= pd.Timestamp(b))
        in_ad = pd.Series(False, index=past.index)
        for a, b in _AD_PERIODS:
            in_ad |= (dates >= pd.Timestamp(a)) & (dates <= pd.Timestamp(b))

        n_act = past.groupby(uid)[d].nunique()
        n_sale = past.loc[in_sale].groupby(uid)[d].nunique()
        n_ad = past.loc[in_ad].groupby(uid)[d].nunique()
        feats["share_activity_in_sales"] = (n_sale / n_act.clip(lower=1)).astype(float)
        feats["share_activity_in_ad"] = (n_ad / n_act.clip(lower=1)).astype(float)

        g_sale = past.loc[in_sale].groupby(uid)[price_col].sum()
        total_gmv = past.groupby(uid)[price_col].sum().clip(lower=1e-9)
        feats["gmv_in_sales_log"] = np.log1p(g_sale).astype(float)
        feats["gmv_sale_ratio"] = (g_sale / total_gmv).astype(float)

    # --- T3.11.A: wake-up (dormant >=30d then order) during sale / gift ------
    if wake_on:
        bought = past[past[price_col] > 0].sort_values([uid, d])
        if len(bought):
            gap = bought.groupby(uid)[d].diff().dt.days
            woke = (gap >= _DORMANT_DAYS) & pd.notna(gap)
            w = bought.loc[woke, [uid]].copy()
            wd = bought.loc[woke, d]
            w["in_sale"] = False
            for a, b in _SALES_2025 + _SALES_2026:
                w.loc[(wd >= pd.Timestamp(a)) & (wd <= pd.Timestamp(b)), "in_sale"] = True
            w["in_gift"] = (
                (wd >= pd.Timestamp("2025-02-14")) & (wd <= pd.Timestamp("2025-03-15"))
            ).to_numpy()
            feats["woke_up_during_sale"] = w.groupby(uid)["in_sale"].sum().astype(float)
            feats["woke_up_during_gift"] = w.groupby(uid)["in_gift"].sum().astype(float)

    # --- T3.11.B: frequency acceleration (orders last 90d / prior 90d) -------
    if freq_on and "to_ord" in past.columns:
        cut90 = ref_date - pd.Timedelta(days=90)
        cut180 = ref_date - pd.Timedelta(days=180)
        recent = past[(dates > cut90) & (dates <= ref_date)]
        prior = past[(dates > cut180) & (dates <= cut90)]
        r_ord = recent.groupby(uid)["to_ord"].sum()
        p_ord = prior.groupby(uid)["to_ord"].sum()
        feats["freq_accel_90v180"] = (r_ord / p_ord.clip(lower=1.0)).astype(float)
        feats["orders_90d"] = r_ord.astype(float)
    return feats


def build_features(
    df: pd.DataFrame,
    ref_date: pd.Timestamp | str | None = None,
    user_id_col: str = "user_id",
    date_col: str = "event_date",
    price_col: str = "price",
    rolling_windows: tuple[int, ...] | None = None,
) -> pd.DataFrame:
    """Build per-user feature DataFrame for LTV forecasting.

    Vectorized implementation: all features are computed with pandas groupby
    aggregations (a few vectorized passes) instead of a per-user Python loop.
    Semantics are identical to the original O(N_users x N_rows) loop, but the
    full 30.6M-row train builds in seconds instead of hours.

    Args:
        df: transaction-level DataFrame with user_id, event_date, price.
        ref_date: reference timestamp. Features use only data before
            ref_date (no future leakage). Default: max(event_date).
        user_id_col: name of user identifier column.
        date_col: name of date column.
        price_col: name of price/GMV column.
        rolling_windows: tuple of window sizes in days.

    Returns:
        DataFrame with one row per user_id, columns:
        - user_id
        - Calendar: day_of_week, month, quarter, is_weekend, is_holiday_ru
        - Rolling: gmv_avg_{w}d, orders_avg_{w}d (for each w in windows),
          gmv_max_30d, n_active_days_30d
        - RFM: frequency, recency, T, monetary_value
        - Cross: recency_x_frequency, recency_x_avg_order_value, frequency_x_monetary
        - Tenure: days_since_first_purchase, is_new_customer
    """
    if ref_date is None:
        ref_date = df[date_col].max()
    ref_date = pd.Timestamp(ref_date)

    # T3.9.2: rolling-window axis A/B - ECUP26_ROLLING_WINDOWS="7,14,30,60,90,120"
    # overrides the default windows (env, dev-only, like the walk-forward config).
    if rolling_windows is None:
        env = os.environ.get("ECUP26_ROLLING_WINDOWS")
        if env and env.strip():
            rolling_windows = tuple(int(x) for x in env.split(","))
        else:
            rolling_windows = DEFAULT_ROLLING_WINDOWS

    # Normalize event dates once. The source parquet stores date32, which
    # pandas reads as object datetime.date; datetime64 makes comparisons and
    # day-difference arithmetic fully vectorized.
    work = df.copy()
    work[date_col] = pd.to_datetime(work[date_col])

    past = work[work[date_col] <= ref_date]
    if past.empty:
        return pd.DataFrame(columns=_feature_columns(rolling_windows))

    uid, d, p = user_id_col, date_col, price_col

    # Per-user aggregates: first/last event date, n_orders, price stats.
    date_agg = past.groupby(uid)[d].agg(["min", "max"])
    price_agg = past.groupby(uid)[p].agg(["sum", "mean", "max"])
    n_orders = past.groupby(uid).size()

    first_date = date_agg["min"]
    last_date = date_agg["max"]
    monetary = price_agg["mean"].astype(float)

    # Mode event date per user = smallest date among the most frequent ones.
    # date_counts' index is sorted (user_id, date), so idxmax() picks the
    # smallest date on ties -- matching pd.Series.mode().iloc[0].
    date_counts = past.groupby([uid, d]).size()
    mode_idx = date_counts.groupby(level=0).idxmax()
    mode_date = pd.Series([idx[1] for idx in mode_idx], index=mode_idx.index, name="_mode_date")

    # Rolling windows: gmv/orders averages over the last 7/14/30 days.
    win30: pd.DataFrame | None = None
    roll: dict[str, pd.Series] = {}
    for w in rolling_windows:
        cutoff = ref_date - pd.Timedelta(days=w)
        win = past[past[d] > cutoff]
        if w == 30:
            win30 = win
        roll[f"gmv_avg_{w}d"] = win.groupby(uid)[p].sum() / w
        roll[f"orders_avg_{w}d"] = win.groupby(uid).size() / w
    if win30 is None:
        cutoff30 = ref_date - pd.Timedelta(days=30)
        win30 = past[past[d] > cutoff30]
    gmv_max_30d = win30.groupby(uid)[p].max()
    active_days_30d = win30.groupby(uid)[d].nunique()

    # Derived RFM / cross / tenure features.
    frequency = (n_orders - 1).clip(lower=0).astype(float)
    recency = (last_date - first_date).dt.days.astype(float)
    ref_series = pd.Series(ref_date, index=first_date.index)
    T = (ref_series - first_date).dt.days.astype(float)
    recency_inv = (T - recency).clip(lower=0.0)  # days since last event

    # Rich activity features from the full daily schema (when columns present).
    activity = _activity_features(past, uid, d, ref_date, rolling_windows)

    # T3.14/WS-1: history-length-invariant (normalized) features.
    # The gift-season training folds (ECUP26_EXTRA_ORIGINS=2025-02-13, -02-23,
    # -03-08) have SHORT history (~6-9 weeks) while the production infer users
    # have ~13 months. Absolute totals (frequency, monetary, T, gmv) are NOT
    # comparable across such origins, so the gift-season label does not
    # transfer. Per-day rates are history-invariant: a 6-week user buying on
    # every active day and a 13-month user buying on every active day both have
    # activity_rate ~1.0 / orders_per_active_day ~N, so the gift surge learned
    # on short-history folds maps onto the full-history test users.
    # Env-gated (ECUP26_HIST_NORM_FEATURES, default ON) for ablation A/B.
    if os.environ.get("ECUP26_HIST_NORM_FEATURES", "1") == "1":
        n_active_total = past.groupby(uid)[d].nunique().astype(float)
        days_alive = T.clip(lower=1.0)
        total_gmv = price_agg["sum"].astype(float)
        if "to_ord" in past.columns:
            total_orders = past.groupby(uid)["to_ord"].sum().astype(float)
        else:
            # Synthetic/no-order-schema fallback: rows == active days.
            total_orders = n_active_total.copy()
        activity["orders_per_active_day"] = total_orders / n_active_total.clip(lower=1.0)
        activity["gmv_per_active_day"] = total_gmv / n_active_total.clip(lower=1.0)
        activity["activity_rate"] = n_active_total / days_alive
        activity["gmv_per_day_alive"] = total_gmv / days_alive
        activity["orders_per_day_alive"] = total_orders / days_alive
        activity["recency_rate"] = recency_inv / days_alive

    # v3 (T3.8.5): holiday proximity of typical activity + dow of last order.
    activity["days_to_nearest_holiday"] = mode_date.map(_days_to_nearest_holiday).astype(float)
    if "to_ord" in past.columns:
        last_ord = past[past["to_ord"] > 0].groupby(uid)[d].max()
        activity["dow_last_order"] = last_ord.dt.dayofweek.astype(float)
    else:
        activity["dow_last_order"] = last_date.dt.dayofweek.astype(float)

    # v3.2 (T3.8.7): Russian/Ozon seasonal, promo and gift features.
    activity.update(_seasonal_features(past, uid, d, ref_date, p))

    # T3.10.4: absence/gap features (env-gated; dev-only A/B, OFF by default).
    if os.environ.get("ECUP26_ABSENCE_FEATURES", "0") == "1":
        activity.update(_absence_features(past, uid, d, ref_date))
    # T3.10.12: behavioral purchase-timing traits (env-gated A/B, OFF by default).
    if os.environ.get("ECUP26_BEHAVIOR_FEATURES", "0") == "1":
        activity.update(_behavioral_features(past, uid, d, p))
    # T3.10.17: actor-type features (env-gated A/B, OFF by default).
    if os.environ.get("ECUP26_ACTOR_FEATURES", "0") == "1":
        activity.update(_actor_features(past, uid, d, p))
    # T3.10.15: gift-holiday pre/post purchase windows (env-gated A/B, OFF by default).
    if os.environ.get("ECUP26_HOLIDAY_PREPOST_FEATURES", "0") == "1":
        activity.update(_holiday_prepost_features(past, uid, d, p))
    # T3.11: calendar-v2 features - full sale/ad day flags + wake-up + freq-accel.
    if (
        os.environ.get("ECUP26_CALENDAR_V2_FEATURES", "0") == "1"
        or os.environ.get("ECUP26_CALV2_WAKE", "0") == "1"
        or os.environ.get("ECUP26_CALV2_FREQ", "0") == "1"
        or os.environ.get("ECUP26_CALV2_SALE", "0") == "1"
    ):
        activity.update(_calendar_v2_features(past, uid, d, ref_date, p))
    # T3.16: reactivation / activity-decay features (env-gated A/B, OFF by default).
    if os.environ.get("ECUP26_REACTIVATION_FEATURES", "0") == "1":
        activity.update(_reactivation_features(past, uid, d, p, ref_date))
    # T3.17: per-user Markov state-transition features (env-gated A/B, OFF by default).
    if os.environ.get("ECUP26_MARKOV_FEATURES", "0") == "1":
        activity.update(_markov_state_features(past, uid, d, ref_date))
    # T3.14/Г1.5: precise holiday-day + prediction-window-structure features
    # (env-gated A/B, OFF by default; calv4 candidate).
    if os.environ.get("ECUP26_HOLIDAY_DAY_FEATURES", "0") == "1":
        activity.update(
            _holiday_day_features(past, uid, d, p, ref_date, date_agg.index)
        )

    # Assemble in the canonical column order (index = user_ids for now).
    # Build the whole frame in ONE pd.DataFrame(...) call (T3.8.11):
    # column-by-column assignment fragments pandas and is slow/memory-heavy
    # on 30M rows (PerformanceWarning). Values are unchanged.
    cols: dict[str, pd.Series] = {
        "day_of_week": mode_date.dt.dayofweek.astype(int),
        "month": mode_date.dt.month.astype(int),
        "quarter": ((mode_date.dt.month - 1) // 3 + 1).astype(int),
        "is_weekend": (mode_date.dt.dayofweek >= 5).astype(int),
        "is_holiday_ru": mode_date.map(_is_holiday).astype(int),
    }
    for name, s in roll.items():
        cols[name] = s.reindex(date_agg.index).fillna(0.0).astype(float)
    cols["gmv_max_30d"] = gmv_max_30d.reindex(date_agg.index).fillna(0.0).astype(float)
    cols["n_active_days_30d"] = active_days_30d.reindex(date_agg.index).fillna(0.0).astype(float)
    # T3.9.11: small feature gaps from the zero/whale error analysis.
    # 1) active-day counts at finer windows (only 30d existed).
    for w, name in ((7, "n_active_days_7d"), (14, "n_active_days_14d")):
        cw = ref_date - pd.Timedelta(days=w)
        cols[name] = past[past[d] > cw].groupby(uid)[d].nunique().reindex(date_agg.index).fillna(0.0).astype(float)
    # 2) abandoned cart in the last 30d (cart activity without an order).
    if "to_cart_30d" in activity and "to_ord_30d" in activity:
        cols["has_cart_no_order_30d"] = ((activity["to_cart_30d"] > 0) & (activity["to_ord_30d"] == 0)).astype(float)
    # 3) older-history window: GMV over (ref-180, ref-90] (beyond the 90d peak).
    cw_old = ref_date - pd.Timedelta(days=180)
    cw_mid = ref_date - pd.Timedelta(days=90)
    cols["gmv_90_180d"] = past[(past[d] > cw_old) & (past[d] <= cw_mid)].groupby(uid)[p].sum().reindex(date_agg.index).fillna(0.0).astype(float)
    cols["frequency"] = frequency.reindex(date_agg.index).fillna(0.0)
    cols["recency"] = recency.reindex(date_agg.index).fillna(0.0)
    cols["T"] = T.reindex(date_agg.index).fillna(0.0)
    cols["monetary_value"] = monetary.reindex(date_agg.index).fillna(0.0)
    cols["recency_x_frequency"] = (recency_inv * frequency).reindex(date_agg.index).fillna(0.0)
    cols["recency_x_avg_order_value"] = (recency_inv * monetary).reindex(date_agg.index).fillna(0.0)
    cols["frequency_x_monetary"] = (frequency * monetary).reindex(date_agg.index).fillna(0.0)
    cols["days_since_first_purchase"] = T.reindex(date_agg.index).fillna(0.0)
    cols["is_new_customer"] = (T < 7).astype(float).reindex(date_agg.index).fillna(0.0)
    for name, s in activity.items():
        fill = _NEVER_ACTIVE_RECENCY if (name.startswith("recency_") and not name.endswith("_ratio")) else 0.0
        cols[name] = s.reindex(date_agg.index).fillna(fill).astype(float)

    out = pd.DataFrame(cols, index=date_agg.index)
    out.index.name = uid

    # Promote user_id from the index to the first column.
    return out.reset_index()
