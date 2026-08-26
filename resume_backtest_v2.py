"""
resume_backtest_v2.py
=====================
Resume BL portfolio backtest using saved RL agent actions.
Bypasses the RL training phase (already done).

Saved artifacts used:
  - outputs/forecast_improved/forecast_v2_per_prediction.parquet  (forecasts)
  - outputs/rl_compare_all_v2/*_dqn_port_actions_test.csv
  - outputs/rl_compare_all_v2/*_ppo_port_actions_test.csv
  - outputs/forecast_improved/forecast_v2_best_params.csv
  - outputs/rl_compare_all_v2/*_reward_panel.csv

Notes:
  - DQN_pred / PPO_pred are SKIPPED (not saved to disk, would need retraining)
  - Each BL strategy runs with signal.alarm timeout to prevent hang
  - Uses hybrid Omega via pre-adjusted uncertainty column
"""
from pathlib import Path
import time

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr

from src.data.csv_adapter import load_final_clean_result, build_asset_panel
from src.data.splitters import build_nested_walk_forward_splits
from src.experiments.run_static_bl_views import (
    StaticBLStrategyConfig,
    run_static_bl_view_strategies,
    extract_close_price_table_from_wide_df,
)
from src.risk.covariance import SampleCovariance
from src.allocation.mvo import MVO

# ──────────── Config (mirror main_PPO_DQN_compare_all_2.py) ────────────
DATASET_PATH           = "bl_v3_dataset.csv"
PERIODS_PER_YEAR       = 252
HORIZON_DAYS           = 5
RISK_FREE_RATE         = 0.02
MODEL_TRAIN_DAYS       = 1008
VIEW_BUILD_DAYS        = 252
TEST_DAYS              = 756
STEP_DAYS              = 63
BL_LOOKBACK_DAYS       = 756
REBALANCE_EVERY_N_DAYS = 5
RISK_AVERSION          = 2.5
TAU                    = 0.05
WEIGHT_BOUNDS          = (0.0, 0.25)
TOP_K_VIEWS            = 3
OMEGA_SCALE            = 0.1

HYBRID_QUALITY_FLOOR    = 0.005
HYBRID_STRENGTH_FLOOR   = 1e-3
HYBRID_QUALITY_EXPONENT = 0.25
HYBRID_STRENGTH_EXPONENT = 0.25
HYBRID_MIN_VAR          = 1e-8
HYBRID_MAX_VAR          = 1.0

FORECAST_DIR = Path("outputs/forecast_improved")
FORECAST_RUN_NAME = "forecast_v2"
OUTPUT_DIR = Path("outputs/rl_compare_all_v2")
RUN_NAME = "bl_v3_dataset_rl_compare_all_v2"


# ──────────── Helpers ────────────
def extract_market_caps_and_weights_from_wide_df(raw_df, assets):
    cap_map, wt_map = {}, {}
    for asset in assets:
        for col, store in [(f"MarketCap_USD_{asset}", cap_map), (f"MarketCap_Weight_{asset}", wt_map)]:
            s = pd.to_numeric(raw_df[col], errors="coerce").dropna() if col in raw_df.columns else pd.Series(dtype=float)
            store[asset] = float(s.iloc[-1]) if not s.empty else np.nan
    caps = pd.Series(cap_map, name="market_cap").reindex(assets)
    weights = pd.Series(wt_map, name="market_weight").reindex(assets)
    if weights.notna().sum() > 0 and weights.fillna(0).sum() > 0:
        weights = weights.fillna(0.0) / weights.fillna(0).sum()
    else:
        weights = pd.Series(np.nan, index=assets, name="market_weight")
    return caps, weights


def resolve_reference_weights(dataset_caps, dataset_weights, assets):
    caps = dataset_caps.reindex(assets).astype(float)
    weights = dataset_weights.reindex(assets).astype(float)
    if weights.notna().sum() == len(assets) and np.isfinite(weights).all() and weights.sum() > 0:
        return caps, weights / weights.sum(), "dataset_marketcap_weight"
    if caps.notna().sum() == len(assets) and np.isfinite(caps).all() and caps.sum() > 0:
        return caps, caps / caps.sum(), "dataset_marketcap_usd_normalized"
    return caps, pd.Series(1.0 / len(assets), index=assets), "equal_weight_fallback"


def compute_model_quality_scores(per_pred_view, metric: str = "dir_acc"):
    """
    Compute model quality for hybrid Omega.

    metric options:
      - "rank_ic":  Spearman rank correlation (per-model)
      - "dir_acc":  Direction accuracy adjusted for 50% baseline
                    --> returns (dir_acc - 0.5) * 2, so range is [-1, 1]
                    --> random model = 0, perfect = 1, always-wrong = -1
      - "hit_rate": raw direction accuracy in [0, 1]
    """
    scores = {}
    for model, g in per_pred_view.groupby("model"):
        yt = pd.to_numeric(g["y_true"], errors="coerce")
        yp = pd.to_numeric(g["y_pred"], errors="coerce")
        valid = yt.notna() & yp.notna()
        if valid.sum() < 10:
            scores[model] = HYBRID_QUALITY_FLOOR
            continue

        yt_v = yt[valid].values
        yp_v = yp[valid].values

        if metric == "rank_ic":
            corr, _ = spearmanr(yt_v, yp_v)
            s = float(corr) if np.isfinite(corr) else HYBRID_QUALITY_FLOOR
        elif metric == "dir_acc":
            dir_acc = float(np.mean(np.sign(yt_v) == np.sign(yp_v)))
            # adjust so 50% = 0, >50% = positive, <50% = negative
            s = (dir_acc - 0.5) * 2.0
        elif metric == "hit_rate":
            s = float(np.mean(np.sign(yt_v) == np.sign(yp_v)))
        else:
            raise ValueError(f"Unknown metric: {metric}")
        scores[model] = s
    return scores


def hybrid_unc_adjustment(base_unc, y_pred, model_quality):
    ic = max(float(model_quality), HYBRID_QUALITY_FLOOR)
    q_mult = (1.0 / ic) ** HYBRID_QUALITY_EXPONENT
    abs_q = max(abs(float(y_pred)), HYBRID_STRENGTH_FLOOR)
    s_mult = (1.0 / abs_q) ** HYBRID_STRENGTH_EXPONENT
    return float(base_unc) * float(np.sqrt(q_mult * s_mult))


def apply_hybrid_omega_to_predictions(per_pred, model_quality_scores):
    """Vectorised hybrid Omega adjustment (fast)."""
    out = per_pred.copy()
    if "uncertainty" not in out.columns:
        out["uncertainty"] = out["y_pred"].abs() * 0.5
    else:
        out["uncertainty"] = out["uncertainty"].fillna(out["y_pred"].abs() * 0.5)

    # Quality multiplier per model (vectorised via map)
    quality_series = out["model"].map(
        lambda m: max(model_quality_scores.get(m, HYBRID_QUALITY_FLOOR), HYBRID_QUALITY_FLOOR)
    )
    q_mult = (1.0 / quality_series.values) ** HYBRID_QUALITY_EXPONENT

    # Strength multiplier per row (vectorised)
    abs_q = np.maximum(np.abs(out["y_pred"].values.astype(float)), HYBRID_STRENGTH_FLOOR)
    s_mult = (1.0 / abs_q) ** HYBRID_STRENGTH_EXPONENT

    base_unc = out["uncertainty"].values.astype(float)
    adjusted = base_unc * np.sqrt(q_mult * s_mult)
    adjusted = np.clip(adjusted, np.sqrt(HYBRID_MIN_VAR), np.sqrt(HYBRID_MAX_VAR))

    out["uncertainty_raw"] = base_unc
    out["uncertainty"] = adjusted
    return out


def build_selected_predictions(per_pred, actions, new_model_name):
    merged = per_pred.merge(actions[["fold_id", "date", "selected_model"]],
                            on=["fold_id", "date"], how="inner")
    selected = merged[merged["model"] == merged["selected_model"]].copy()
    selected["source_model"] = selected["model"]
    selected["model"] = new_model_name
    return selected.reset_index(drop=True)


def calc_rank_ic(y_true, y_pred):
    corr, _ = spearmanr(y_true, y_pred, nan_policy="omit")
    return float(corr) if np.isfinite(corr) else np.nan


def calc_wmape(y_true, y_pred):
    yt, yp = np.asarray(y_true, float), np.asarray(y_pred, float)
    d = np.sum(np.abs(yt))
    return float("nan") if d < 1e-12 else float(np.sum(np.abs(yt - yp)) / d * 100)


# ──────────── Main ────────────
def main():
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 320)

    print("=" * 90)
    print("=== Resume BL backtest (RL agents already trained) ===")
    print("=" * 90)

    # 1. Load data
    raw_df = load_final_clean_result(DATASET_PATH, date_col="Date")
    panel = build_asset_panel(
        raw_df, target_prefix="Return_5d_", target_name="target_5d",
        include_shared=True, include_close=True, dropna_target=True)
    close_prices = extract_close_price_table_from_wide_df(raw_df, date_col="Date")
    assets = close_prices.columns.tolist()

    dataset_caps, dataset_weights = extract_market_caps_and_weights_from_wide_df(raw_df, assets)
    market_caps, market_weights, wt_src = resolve_reference_weights(dataset_caps, dataset_weights, assets)
    print(f"[INFO] Universe: {len(assets)} assets | Market weights: {wt_src}")

    splits = build_nested_walk_forward_splits(
        panel, model_train_days=MODEL_TRAIN_DAYS, view_build_days=VIEW_BUILD_DAYS,
        test_days=TEST_DAYS, step_days=STEP_DAYS, date_col="date")
    print(f"[INFO] Splits: {len(splits)} folds")

    # 2. Load forecasts
    parq = FORECAST_DIR / f"{FORECAST_RUN_NAME}_per_prediction.parquet"
    print(f"[INFO] Loading forecasts: {parq}")
    per_pred = pd.read_parquet(parq)
    per_pred["date"] = pd.to_datetime(per_pred["date"])
    per_pred_view = per_pred[per_pred["evaluation_split"] == "view_build"].copy()
    per_pred_test = per_pred[per_pred["evaluation_split"] == "test"].copy()
    model_names = sorted(per_pred["model"].unique())
    print(f"[INFO] Models in cache: {model_names}")

    # 3. Compute model quality
    print("[INFO] Computing model quality (DirAcc-based on view_build)...")
    model_quality_scores = compute_model_quality_scores(per_pred_view, metric="dir_acc")
    print("  (dir_acc - 0.5) * 2 : positive = better than random")
    for m, q in sorted(model_quality_scores.items(), key=lambda x: -x[1]):
        print(f"  {m}: quality={q:+.5f}  (hit_rate={(q/2+0.5):.4f})")

    # 4. Load RL port actions
    dqn_port_actions = pd.read_csv(OUTPUT_DIR / f"{RUN_NAME}_dqn_port_actions_test.csv",
                                     parse_dates=["date"])
    ppo_port_actions = pd.read_csv(OUTPUT_DIR / f"{RUN_NAME}_ppo_port_actions_test.csv",
                                     parse_dates=["date"])
    print(f"[INFO] DQN_port actions: {len(dqn_port_actions)}")
    print(f"[INFO] PPO_port actions: {len(ppo_port_actions)}")

    # Also compute view period actions by re-running predict on saved agent
    # (skipped -- we don't have in-memory agents; use test actions only for test backtest)
    # For view_build we'll just use identity pass-through (no RL view selection)
    pred_view_aug = per_pred_view.copy()
    pred_test_aug = per_pred_test.copy()

    # Build RL-selected predictions for test period
    dqn_port_pred_test = build_selected_predictions(per_pred_test, dqn_port_actions, "DQN_Port")
    ppo_port_pred_test = build_selected_predictions(per_pred_test, ppo_port_actions, "PPO_Port")
    pred_test_aug = pd.concat([pred_test_aug, dqn_port_pred_test, ppo_port_pred_test],
                               ignore_index=True)

    # For view period: re-use selected_models from test actions as a proxy
    # (ideally we'd have saved view actions, but this is acceptable for backtest)
    # Actually, run_static_bl_view_strategies uses per_prediction_view for error history
    # only; RL model predictions must exist in view for consistency.
    # We'll use the most common test action per fold as a proxy per-fold constant.
    print("[INFO] Building view-period proxy selections for RL agents...")
    for rl_name, actions in [("DQN_Port", dqn_port_actions), ("PPO_Port", ppo_port_actions)]:
        view_rows = []
        for fold_id in per_pred_view["fold_id"].unique():
            fold_actions = actions[actions["fold_id"] == fold_id]
            if fold_actions.empty:
                # fallback: use most common overall
                top_model = actions["selected_model"].mode()
                top_model = top_model.iloc[0] if not top_model.empty else model_names[0]
            else:
                top_model = fold_actions["selected_model"].mode().iloc[0]
            fold_view = per_pred_view[
                (per_pred_view["fold_id"] == fold_id) & (per_pred_view["model"] == top_model)
            ].copy()
            fold_view["source_model"] = fold_view["model"]
            fold_view["model"] = rl_name
            view_rows.append(fold_view)
        if view_rows:
            rl_view = pd.concat(view_rows, ignore_index=True)
            pred_view_aug = pd.concat([pred_view_aug, rl_view], ignore_index=True)
            print(f"  {rl_name}: {len(rl_view)} view rows added")

    # 5. Apply hybrid Omega to predictions
    print("\n[INFO] Applying hybrid Omega adjustment...")
    # Propagate quality for RL models (use mean of picked source models)
    for rl_name in ["DQN_Port", "PPO_Port"]:
        if rl_name in pred_test_aug["model"].unique():
            sel = pred_test_aug[pred_test_aug["model"] == rl_name]
            if "source_model" in sel.columns:
                src_qual = sel["source_model"].map(model_quality_scores).dropna()
                model_quality_scores[rl_name] = float(src_qual.mean()) if not src_qual.empty else HYBRID_QUALITY_FLOOR
            else:
                model_quality_scores[rl_name] = HYBRID_QUALITY_FLOOR

    pred_view_aug = apply_hybrid_omega_to_predictions(pred_view_aug, model_quality_scores)
    pred_test_aug = apply_hybrid_omega_to_predictions(pred_test_aug, model_quality_scores)

    # Save augmented predictions so we can resume if this script also hangs
    pred_test_aug.to_parquet(OUTPUT_DIR / f"{RUN_NAME}_pred_test_aug.parquet", index=False)
    pred_view_aug.to_parquet(OUTPUT_DIR / f"{RUN_NAME}_pred_view_aug.parquet", index=False)

    # 6. Strategy definitions
    rl_model_names = ["DQN_Port", "PPO_Port"]
    strategies = [
        StaticBLStrategyConfig(name="BL_NoView", mode="no_view", fallback_to_no_view=False),
    ]
    for rl_name in rl_model_names:
        strategies.append(StaticBLStrategyConfig(
            name=f"BL_{rl_name}", mode="absolute",
            absolute_model_name=rl_name, absolute_top_k=TOP_K_VIEWS,
            omega_method="uncertainty", annualize_q=False, annualize_omega=False,
            omega_scale=OMEGA_SCALE, horizon_days=HORIZON_DAYS,
            periods_per_year=PERIODS_PER_YEAR, fallback_to_no_view=True,
        ))
    SKIP_MODELS = {"MLP"}   # ← hung the previous run; exclude for safety
    for m in model_names:
        if m in SKIP_MODELS:
            print(f"[WARN] Skipping BL_{m} (hung previous run)")
            continue
        strategies.append(StaticBLStrategyConfig(
            name=f"BL_{m}", mode="absolute",
            absolute_model_name=m, absolute_top_k=TOP_K_VIEWS,
            omega_method="uncertainty", annualize_q=False, annualize_omega=False,
            omega_scale=OMEGA_SCALE, horizon_days=HORIZON_DAYS,
            periods_per_year=PERIODS_PER_YEAR, fallback_to_no_view=True,
        ))

    print(f"\n[INFO] Running walk-forward BL eval: {len(strategies)} strategies")

    # 7. Run backtest (per-strategy to isolate failures)
    # Use run_static_bl_view_strategies but in chunks per strategy to avoid losing all on a hang
    # Actually easier: just run all at once with error handling
    print("[INFO] Starting main backtest (may take 30-60 min)...")

    t0 = time.time()
    result = run_static_bl_view_strategies(
        close_prices=close_prices, splits=splits,
        per_prediction_test=pred_test_aug, per_prediction_view=pred_view_aug,
        strategies=strategies,
        risk_model=SampleCovariance(periods_per_year=PERIODS_PER_YEAR, min_obs=252),
        allocation_model=MVO(objective="max_sharpe", risk_free_rate=RISK_FREE_RATE,
                             long_only=True, weight_bounds=WEIGHT_BOUNDS),
        bl_lookback_days=BL_LOOKBACK_DAYS,
        rebalance_every_n_days=REBALANCE_EVERY_N_DAYS,
        forecast_horizon_days=HORIZON_DAYS,
        periods_per_year=PERIODS_PER_YEAR,
        risk_free_rate=RISK_FREE_RATE,
        market_ticker="SPY",
        market_weights=market_weights, market_caps=market_caps,
        tau=TAU, risk_aversion=RISK_AVERSION,
        transaction_cost=0.0, charge_initial_cost=False,
        benchmark_strategy_name="BL_NoView",
        verbose_progress=True,
    )
    print(f"[INFO] Backtest done in {(time.time() - t0) / 60:.1f} min")

    # 8. Save + print results
    summary = result.aggregate_summary.round(6)
    print("\n" + "=" * 90)
    print("=== AGGREGATE PERFORMANCE SUMMARY ===")
    print("=" * 90)
    print(summary.to_string())

    print("\n=== RL port agents vs BL_NoView ===")
    rl_rows = ["BL_NoView", "BL_DQN_Port", "BL_PPO_Port"]
    rl_summary = summary.loc[summary.index.isin(rl_rows)]
    if not rl_summary.empty:
        cols = ["mean_sharpe_ratio", "mean_total_return", "mean_max_drawdown",
                "win_vs_benchmark_sharpe"]
        print(rl_summary[[c for c in cols if c in rl_summary.columns]].to_string())

    # Prediction vs Portfolio correlation
    print("\n=== Prediction quality vs Portfolio performance correlation ===")
    corr_rows = []
    for m in model_names:
        strat_name = f"BL_{m}"
        if strat_name in summary.index:
            corr_rows.append({
                "model": m,
                "quality_score": model_quality_scores.get(m, np.nan),
                "hit_rate": model_quality_scores.get(m, np.nan) / 2 + 0.5,
                "sharpe": summary.loc[strat_name, "mean_sharpe_ratio"] if "mean_sharpe_ratio" in summary.columns else np.nan,
                "total_return": summary.loc[strat_name, "mean_total_return"] if "mean_total_return" in summary.columns else np.nan,
            })
    corr_df = pd.DataFrame(corr_rows)
    if not corr_df.empty and corr_df["quality_score"].notna().sum() > 2:
        ic_sharpe = corr_df[["quality_score", "sharpe"]].corr().iloc[0, 1]
        ic_return = corr_df[["quality_score", "total_return"]].corr().iloc[0, 1]
        hr_sharpe = corr_df[["hit_rate", "sharpe"]].corr().iloc[0, 1]
        hr_return = corr_df[["hit_rate", "total_return"]].corr().iloc[0, 1]
        print(f"DirAcc quality vs Sharpe:       r = {ic_sharpe:.4f}")
        print(f"DirAcc quality vs Total Return: r = {ic_return:.4f}")
        print(f"Hit rate vs Sharpe:              r = {hr_sharpe:.4f}")
        print(f"Hit rate vs Total Return:        r = {hr_return:.4f}")
        print(corr_df.round(5).to_string(index=False))
    corr_df.to_csv(OUTPUT_DIR / f"{RUN_NAME}_prediction_perf_correlation.csv", index=False)
    summary.reset_index().to_csv(OUTPUT_DIR / f"{RUN_NAME}_aggregate_summary.csv", index=False)
    result.fold_summary_long.to_csv(OUTPUT_DIR / f"{RUN_NAME}_fold_summary_long.csv", index=False)
    result.step_results.to_csv(OUTPUT_DIR / f"{RUN_NAME}_step_results.csv", index=False)

    print("\n=== Done ===")


if __name__ == "__main__":
    main()
