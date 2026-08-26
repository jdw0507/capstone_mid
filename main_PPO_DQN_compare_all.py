from __future__ import annotations

"""
main_PPO_DQN_compare_all.py
============================
4 RL model-selectors × BL portfolio walk-forward comparison.

Agents compared:
  1. DQN_pred  — per-asset selection, reward = prediction accuracy
  2. DQN_port  — cross-sectional selection, reward = EMA-Sharpe (BL return)
  3. PPO_pred  — per-asset selection, reward = prediction accuracy
  4. PPO_port  — cross-sectional selection, reward = EMA-Sharpe (BL return)

Plus benchmarks:
  - BL_NoView  (market-implied only)
  - BL_<model>  (each individual ML/DL model)
"""

from pathlib import Path
import time

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler

# ── Data / forecasting infrastructure ────────────────────────────────────────
from src.data.csv_adapter import load_final_clean_result, build_asset_panel
from src.data.splitters import build_nested_walk_forward_splits, describe_split_coverage
from src.forecasting.targets import add_excess_target_to_panel
from src.forecasting.prediction_store import save_forecast_artifacts

from src.forecasting.ml.linear import LinearForecaster
from src.forecasting.ml.decision_tree import DecisionTreeForecaster
from src.forecasting.ml.random_forest import RandomForestForecaster
from src.forecasting.ml.xgboost import XGBoostForecaster
from src.forecasting.ml.lightgbm import LightGBMForecaster
from src.forecasting.ml.catboost import CatBoostForecaster
from src.forecasting.dl.patchtst import PatchTSTForecaster
from src.forecasting.dl.cnn1d import CNN1DForecaster
from src.forecasting.dl.lstm import LSTMForecaster
from src.forecasting.dl.mlp import MLPForecaster
from src.forecasting.dl.transformer import TransformerForecaster
from src.forecasting.dl.hybrid_lstm_transformer import HybridLSTMTransformerForecaster

from src.experiments.run_forecast_baselines import run_forecast_baselines_on_splits
from src.experiments.run_static_bl_views import (
    StaticBLStrategyConfig,
    run_static_bl_view_strategies,
    extract_close_price_table_from_wide_df,
)
from src.expected_returns.black_litterman import BlackLittermanExpectedReturn
from src.risk.covariance import SampleCovariance
from src.allocation.mvo import MVO

# ── RL agents ────────────────────────────────────────────────────────────────
from src.forecasting.rl.DQN_pred import DQNPredSelector, build_dqn_pred_selected_predictions
from src.forecasting.rl.DQN_port import DQNPortConfig, DQNPortSelector
from src.forecasting.rl.PPO_pred import PPOPredSelector, build_ppo_pred_selected_predictions
from src.forecasting.rl.PPO_port import PPOPortConfig, PPOPortSelector


# =============================================================================
# Configuration
# =============================================================================
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
LONG_ONLY              = True
WEIGHT_BOUNDS          = (0.0, 0.25)
TOP_K_VIEWS            = 3
OMEGA_SCALE            = 1.0

FORECAST_CACHE_SUFFIX  = "dqn_compare_reward_modes_stable_mape"
RUN_NAME               = "rl_compare_all"

FORECAST_DIR           = Path("outputs/forecasting")
OUTPUT_DIR             = Path("outputs/rl_compare_all")


# =============================================================================
# Progress logger
# =============================================================================
class ProgressLogger:
    def __init__(self, total: int, desc: str, print_every: int = 1):
        self.total = max(int(total), 1)
        self.desc = desc
        self.print_every = max(int(print_every), 1)
        self.count = 0
        self.start = time.time()

    def update(self, n: int = 1, extra: str = "") -> None:
        self.count += n
        if self.count % self.print_every != 0 and self.count != self.total:
            return
        elapsed = time.time() - self.start
        rate = self.count / elapsed if elapsed > 0 else 0.0
        remaining = (self.total - self.count) / rate if rate > 0 else float("nan")
        pct = 100.0 * self.count / self.total
        eta = f"{remaining / 60:.1f}m" if np.isfinite(remaining) else "?"
        print(
            f"[{self.desc}] {self.count}/{self.total} ({pct:5.1f}%) | "
            f"elapsed={elapsed / 60:.1f}m eta={eta}"
            + (f" | {extra}" if extra else "")
        )


# =============================================================================
# Forecaster registry
# =============================================================================
def build_forecasters() -> dict[str, object]:
    kw = dict(target_column="target_5d_excess", horizon_days=HORIZON_DAYS,
              periods_per_year=PERIODS_PER_YEAR)
    return {
        "Linear_ols":            LinearForecaster(linear_type="ols", **kw),
        "Linear_ridge":          LinearForecaster(linear_type="ridge", alpha=1.0, **kw),
        "Linear_lasso":          LinearForecaster(linear_type="lasso", alpha=0.001, **kw),
        "DecisionTree":          DecisionTreeForecaster(max_depth=6, min_samples_leaf=20, **kw),
        "RandomForest":          RandomForestForecaster(n_estimators=400, max_depth=8, min_samples_leaf=10, **kw),
        "XGBoost":               XGBoostForecaster(n_estimators=300, learning_rate=0.03, max_depth=4, **kw),
        "LightGBM":              LightGBMForecaster(n_estimators=400, learning_rate=0.03, num_leaves=31, **kw),
        "CatBoost":              CatBoostForecaster(iterations=400, learning_rate=0.03, depth=6, **kw),
        "PatchTST":              PatchTSTForecaster(lookback=60, patch_len=12, stride=6, epochs=10, batch_size=256, lr=3e-4, dropout=0.1, verbose=False, **kw),
        "CNN1D":                 CNN1DForecaster(lookback=60, channels=(64, 64), kernel_size=3, epochs=10, batch_size=256, lr=1e-3, dropout=0.1, verbose=False, **kw),
        "LSTM":                  LSTMForecaster(lookback=60, hidden_dim=64, num_layers=2, epochs=10, batch_size=256, lr=1e-3, dropout=0.1, verbose=False, **kw),
        "MLP":                   MLPForecaster(lookback=60, hidden_dims=(256, 128), epochs=10, batch_size=256, lr=1e-3, dropout=0.1, verbose=False, **kw),
        "Transformer":           TransformerForecaster(lookback=60, d_model=64, n_heads=4, num_layers=2, epochs=10, batch_size=256, lr=3e-4, dropout=0.1, verbose=False, **kw),
        "HybridLSTMTransformer": HybridLSTMTransformerForecaster(lookback=60, hidden_dim=64, lstm_layers=1, d_model=64, n_heads=4, transformer_layers=1, epochs=10, batch_size=256, lr=3e-4, dropout=0.1, verbose=False, **kw),
    }


# =============================================================================
# Market-weight helpers
# =============================================================================
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


# =============================================================================
# State builder for _port agents (zero look-ahead)
# =============================================================================
def build_state_table(per_prediction_df, model_names):
    rows = []
    for (fold_id, date), g in per_prediction_df.groupby(["fold_id", "date"], sort=True):
        row = {"fold_id": fold_id, "date": pd.Timestamp(date)}
        model_means = []
        for m in model_names:
            preds = g[g["model"] == m]["y_pred"].astype(float)
            pm = float(preds.mean()) if not preds.empty else 0.0
            pa = float(preds.abs().mean()) if not preds.empty else 0.0
            ps = float(preds.std(ddof=0)) if len(preds) > 1 else 0.0
            row[f"pred_mean__{m}"] = pm
            row[f"pred_abs__{m}"] = pa
            row[f"pred_std__{m}"] = ps
            model_means.append(pm)
        arr = np.asarray(model_means, dtype=float)
        row["global_consensus"] = float(arr.mean())
        row["global_dispersion"] = float(arr.std(ddof=0)) if len(arr) > 1 else 0.0
        row["global_abs_mean"] = float(np.abs(arr).mean())
        rows.append(row)

    out = pd.DataFrame(rows).sort_values(["fold_id", "date"]).reset_index(drop=True)
    state_cols = []
    for m in model_names:
        state_cols += [f"pred_mean__{m}", f"pred_abs__{m}", f"pred_std__{m}"]
    state_cols += ["global_consensus", "global_dispersion", "global_abs_mean"]
    out["state"] = out[state_cols].astype(float).fillna(0.0).to_numpy().tolist()
    out["state"] = out["state"].apply(lambda x: np.asarray(x, dtype=np.float32))
    return out[["fold_id", "date", "state"]]


def normalise_states(train_df, test_df):
    train_arr = np.stack(train_df["state"].to_numpy())
    scaler = StandardScaler().fit(train_arr)

    def _apply(df):
        d = df.copy()
        arr = np.stack(d["state"].to_numpy())
        d["state"] = [np.asarray(r, dtype=np.float32) for r in scaler.transform(arr)]
        return d

    return _apply(train_df), _apply(test_df), scaler


# =============================================================================
# BL reward precompute for _port agents (date-safe)
# =============================================================================
def _price_at(close_prices, date):
    if date in close_prices.index:
        return close_prices.loc[date]
    prior = close_prices.index[close_prices.index <= date]
    if len(prior) == 0:
        raise KeyError(f"No price data at or before {date}")
    return close_prices.loc[prior[-1]]


def _future_realized_return(close_prices, d1, d2):
    return (_price_at(close_prices, d2) / _price_at(close_prices, d1) - 1.0).astype(float)


def _train_slice(close_prices, decision_date, lookback=BL_LOOKBACK_DAYS):
    return close_prices.loc[:decision_date].pct_change().dropna().tail(lookback)


def _build_bl_views(per_pred, fold_id, decision_date, model_name, universe,
                    top_k=TOP_K_VIEWS, omega_scale=OMEGA_SCALE):
    snap = per_pred[
        (per_pred["fold_id"] == fold_id)
        & (pd.to_datetime(per_pred["date"]) == pd.Timestamp(decision_date))
        & (per_pred["model"] == model_name)
    ].copy()
    if snap.empty:
        raise ValueError(f"No predictions: fold={fold_id} date={decision_date} model={model_name}")

    # Omega from uncertainty (training residual std) — no look-ahead
    if "uncertainty" in snap.columns and snap["uncertainty"].notna().any():
        snap["_unc"] = snap["uncertainty"].astype(float).fillna(snap["y_pred"].abs() * 0.5)
    else:
        snap["_unc"] = snap["y_pred"].abs() * 0.5

    snap = snap.assign(abs_pred=snap["y_pred"].abs()).nlargest(min(top_k, len(snap)), "abs_pred")

    P, Q, omega_diag = [], [], []
    for _, row in snap.iterrows():
        asset = row["asset"]
        if asset not in universe:
            continue
        vec = np.zeros(len(universe), dtype=float)
        vec[universe.index(asset)] = 1.0
        P.append(vec)
        Q.append(float(row["y_pred"]))                   # Q: view vector
        unc = max(float(row["_unc"]), 1e-6)
        omega_diag.append(max((unc ** 2) * omega_scale, 1e-8))  # Omega diagonal

    if not P:
        raise ValueError("No valid BL views.")
    return (
        np.array(P, dtype=float),       # P: (k, N) pick matrix
        np.array(Q, dtype=float),        # Q: (k,) 1D view vector
        np.diag(omega_diag),             # Omega: (k, k) diagonal uncertainty
    )


def _ensure_1d(x, index, name):
    if isinstance(x, pd.Series):
        return x.copy().reindex(index).rename(name).astype(float)
    if isinstance(x, pd.DataFrame):
        x = x.iloc[:, 0] if x.shape[1] == 1 else x.iloc[0, :]
        return x.reindex(index).rename(name).astype(float)
    return pd.Series(np.asarray(x).reshape(-1), index=index, name=name, dtype=float)


def _compute_bl_portfolio_return(close_prices, per_pred, fold_id, decision_date,
                                  forecast_end_date, model_name, market_caps, market_weights):
    universe = close_prices.columns.tolist()
    train_returns = _train_slice(close_prices, decision_date)
    fallback = float(_future_realized_return(close_prices, decision_date, forecast_end_date).mean())

    if len(train_returns) < 60:
        return fallback

    cov_model = SampleCovariance(periods_per_year=PERIODS_PER_YEAR,
                                  min_obs=min(252, max(20, len(train_returns))))
    cov = cov_model.fit_predict(train_returns).loc[universe, universe].copy()
    np.fill_diagonal(cov.values, cov.values.diagonal() + 1e-6)

    try:
        P, Q, Omega = _build_bl_views(per_pred, fold_id, decision_date, model_name, universe)
        bl = BlackLittermanExpectedReturn(
            tickers=universe, prices=None,
            periods_per_year=PERIODS_PER_YEAR,
            min_obs=min(252, max(20, len(train_returns))),
            tau=TAU, risk_free_rate=RISK_FREE_RATE,
            market_ticker="SPY", market_prices=None,
            market_caps=market_caps.reindex(universe) if market_caps is not None else None,
            market_weights=market_weights.reindex(universe) if market_weights is not None else None,
            risk_aversion=RISK_AVERSION,
            P=P, Q=Q, Omega=Omega, view_confidences=None,
            use_market_cap_weights=True, allow_equal_weight_fallback=True,
            market_weight_fallback="equal_weight", blend_with_equal_weight=None,
        )
        mu = _ensure_1d(bl.fit_predict(train_returns[universe]), universe, "mu")
        mvo = MVO(objective="max_sharpe", risk_free_rate=RISK_FREE_RATE,
                  long_only=LONG_ONLY, weight_bounds=WEIGHT_BOUNDS)
        w = _ensure_1d(mvo.optimize(mu, cov), universe, "w")
    except Exception as e:
        print(f"[WARN] BL fallback | fold={fold_id} date={decision_date.date()} model={model_name} | {e}")
        w = pd.Series(1.0 / len(universe), index=universe, dtype=float)

    realized = _future_realized_return(close_prices, decision_date, forecast_end_date).reindex(universe).fillna(0.0)
    return float(np.dot(w.reindex(universe).fillna(0.0).to_numpy(), realized.to_numpy()))


def build_rebalance_schedule(close_prices, start_date, end_date,
                              rebalance_n=REBALANCE_EVERY_N_DAYS, horizon=HORIZON_DAYS):
    dates = close_prices.index[(close_prices.index >= start_date) & (close_prices.index <= end_date)]
    rows = []
    for i in range(0, len(dates), rebalance_n):
        if i + horizon >= len(dates):
            break
        rows.append({"decision_date": dates[i], "forecast_end_date": dates[i + horizon]})
    return pd.DataFrame(rows)


def build_bl_reward_panel(close_prices, per_pred_view, splits, model_names,
                           market_caps, market_weights):
    all_rows = []
    fold_logger = ProgressLogger(len(splits), "BL reward (folds)", 1)

    for _, sr in splits.iterrows():
        fold_id = int(sr["fold_id"])
        schedule = build_rebalance_schedule(close_prices, pd.Timestamp(sr["view_start"]),
                                             pd.Timestamp(sr["view_end"]))
        n_jobs = max(len(schedule) * len(model_names), 1)
        step_logger = ProgressLogger(n_jobs, f"fold={fold_id}", max(n_jobs // 10, 1))

        for row in schedule.itertuples(index=False):
            d1, d2 = pd.Timestamp(row.decision_date), pd.Timestamp(row.forecast_end_date)
            out_row = {"fold_id": fold_id, "date": d1}
            for m in model_names:
                out_row[f"reward_blret__{m}"] = _compute_bl_portfolio_return(
                    close_prices, per_pred_view, fold_id, d1, d2, m, market_caps, market_weights)
                step_logger.update(1, extra=f"{d1.date()} {m}")
            all_rows.append(out_row)
        fold_logger.update(1, extra=f"fold={fold_id} steps={len(schedule)}")

    return pd.DataFrame(all_rows).sort_values(["fold_id", "date"]).reset_index(drop=True)


# =============================================================================
# Cache loaders
# =============================================================================
def load_or_run_forecasts(dataset_tag, panel, splits, forecasters):
    cache_name = f"{dataset_tag}_{FORECAST_CACHE_SUFFIX}"
    parq = FORECAST_DIR / f"{cache_name}_per_prediction.parquet"
    csv = FORECAST_DIR / f"{cache_name}_per_prediction.csv"

    if parq.exists():
        print(f"[INFO] Loading forecast cache: {parq}")
        df = pd.read_parquet(parq)
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"])
        return df
    if csv.exists():
        print(f"[INFO] Loading forecast cache: {csv}")
        return pd.read_csv(csv, parse_dates=["date"])

    print("[INFO] No forecast cache — running all forecasters…")
    r_view = run_forecast_baselines_on_splits(
        panel_df=panel, splits=splits, forecasters=forecasters,
        target_col="target_5d_excess", evaluation_split="view_build",
        exclude_target_like=True, date_col="date", verbose_progress=True)
    r_test = run_forecast_baselines_on_splits(
        panel_df=panel, splits=splits, forecasters=forecasters,
        target_col="target_5d_excess", evaluation_split="test",
        exclude_target_like=True, date_col="date", verbose_progress=True)
    per_pred = pd.concat([r_view.per_prediction, r_test.per_prediction], ignore_index=True)
    save_forecast_artifacts(
        per_prediction=per_pred,
        summary_by_model=pd.concat([
            r_view.metric_result.summary_by_model.add_suffix("_view"),
            r_test.metric_result.summary_by_model.add_suffix("_test")], axis=1),
        summary_by_model_asset=pd.concat([
            r_view.metric_result.summary_by_model_asset.assign(evaluation_split="view_build"),
            r_test.metric_result.summary_by_model_asset.assign(evaluation_split="test")], ignore_index=True),
        rank_ic_by_date=pd.concat([
            r_view.metric_result.rank_ic_by_date,
            r_test.metric_result.rank_ic_by_date], ignore_index=True),
        output_dir=FORECAST_DIR, run_name=f"{dataset_tag}_{FORECAST_CACHE_SUFFIX}",
        save_csv=True, save_parquet=True)
    return per_pred


def load_or_build_reward_panel(run_name, close_prices, per_pred_view, splits,
                                model_names, market_caps, market_weights):
    path = OUTPUT_DIR / f"{run_name}_reward_panel.csv"
    if path.exists():
        print(f"[INFO] Loading reward panel: {path}")
        return pd.read_csv(path, parse_dates=["date"])
    print("[INFO] Building BL reward panel (slow step)…")
    panel = build_bl_reward_panel(close_prices, per_pred_view, splits,
                                  model_names, market_caps, market_weights)
    panel.to_csv(path, index=False)
    return panel


# =============================================================================
# Post-processing
# =============================================================================
def build_selected_predictions(per_pred, actions, new_model_name):
    merged = per_pred.merge(actions[["fold_id", "date", "selected_model"]],
                            on=["fold_id", "date"], how="inner")
    selected = merged[merged["model"] == merged["selected_model"]].copy()
    selected["source_model"] = selected["model"]
    selected["model"] = new_model_name
    return selected.reset_index(drop=True)


def safe_mape(y_true, y_pred, eps=1e-8):
    yt, yp = np.asarray(y_true, float), np.asarray(y_pred, float)
    return float(np.mean(np.abs((yt - yp) / np.maximum(np.abs(yt), eps))) * 100)


# =============================================================================
# Main experiment
# =============================================================================
def run_experiment(csv_path: str) -> None:
    dataset_tag = Path(csv_path).stem.lower()
    run_name = f"{dataset_tag}_{RUN_NAME}"

    print("\n" + "=" * 90)
    print(f"=== 4-Agent RL Comparison: {csv_path} ===")
    print("=" * 90)

    device_str = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] Device: {device_str}")

    # ── 1. Data ──────────────────────────────────────────────────────────────
    raw_df = load_final_clean_result(csv_path, date_col="Date")
    panel = build_asset_panel(
        raw_df, target_prefix="Return_5d_", target_name="target_5d",
        include_shared=True, include_close=True, dropna_target=True)
    panel = add_excess_target_to_panel(
        panel, target_col="target_5d", output_col="target_5d_excess",
        risk_free_rate=RISK_FREE_RATE, horizon_days=HORIZON_DAYS,
        periods_per_year=PERIODS_PER_YEAR, rf_is_annualized=True)

    close_prices = extract_close_price_table_from_wide_df(raw_df, date_col="Date")
    assets = close_prices.columns.tolist()
    print(f"Universe: {len(assets)} assets")

    dataset_caps, dataset_weights = extract_market_caps_and_weights_from_wide_df(raw_df, assets)
    market_caps, market_weights, wt_src = resolve_reference_weights(dataset_caps, dataset_weights, assets)
    print(f"[INFO] Market weight source: {wt_src}")
    print(f"[INFO] Top-5 weights: {market_weights.nlargest(5).round(4).to_dict()}")

    # ── 2. Splits ────────────────────────────────────────────────────────────
    splits = build_nested_walk_forward_splits(
        panel, model_train_days=MODEL_TRAIN_DAYS, view_build_days=VIEW_BUILD_DAYS,
        test_days=TEST_DAYS, step_days=STEP_DAYS, date_col="date")
    print(f"\nSplits: {len(splits)} folds")
    print(describe_split_coverage(splits).to_string(index=False))

    # ── 3. Forecasting ──────────────────────────────────────────────────────
    forecasters = build_forecasters()
    model_names = list(forecasters.keys())
    print(f"\nForecasting models ({len(model_names)}): {model_names}")

    per_pred = load_or_run_forecasts(dataset_tag, panel, splits, forecasters)
    per_pred_view = per_pred[per_pred["evaluation_split"] == "view_build"].copy()
    per_pred_test = per_pred[per_pred["evaluation_split"] == "test"].copy()

    mape_rows = []
    for m, g in per_pred.groupby("model"):
        mape_rows.append({"model": m, "mape_pct": safe_mape(g["y_true"], g["y_pred"])})
    mape_df = pd.DataFrame(mape_rows).sort_values("mape_pct")
    print("\n=== MAPE by model ===")
    print(mape_df.round(3).to_string(index=False))

    # ── 4. State tables (_port agents) ───────────────────────────────────────
    print("\n[INFO] Building state tables for _port agents…")
    raw_train_st = build_state_table(per_pred_view, model_names)
    raw_test_st = build_state_table(per_pred_test, model_names)
    train_states, test_states, _ = normalise_states(raw_train_st, raw_test_st)
    print(f"[INFO] State dim={len(train_states.iloc[0]['state'])} | "
          f"train={len(train_states)} test={len(test_states)}")

    # ── 5. BL reward panel (_port agents) ─────────────────────────────────
    reward_panel = load_or_build_reward_panel(
        run_name, close_prices, per_pred_view, splits,
        model_names, market_caps, market_weights)
    print(f"[INFO] Reward panel: {len(reward_panel)} rows")

    # ══════════════════════════════════════════════════════════════════════════
    # 6. Train all 4 RL agents
    # ══════════════════════════════════════════════════════════════════════════
    pred_view_aug = per_pred_view.copy()
    pred_test_aug = per_pred_test.copy()

    # ── 6a. DQN_pred ─────────────────────────────────────────────────────────
    print("\n" + "─" * 90)
    print("=== Agent 1/4: DQN_pred (per-asset, prediction accuracy) ===")
    print("─" * 90)
    dqn_pred_selected, dqn_pred_diag = build_dqn_pred_selected_predictions(
        per_prediction=per_pred,
        candidate_models=model_names,
        train_split="view_build",
        infer_splits=("view_build", "test"),
        output_model_name="DQN_Pred",
        params=dict(hidden_dims=(128, 64), gamma=0.95, lr=1e-3, episodes=12,
                    device=device_str, verbose=True),
        verbose=True,
    )
    dqn_pred_diag.to_csv(OUTPUT_DIR / f"{run_name}_dqn_pred_diagnostics.csv", index=False)
    if not dqn_pred_selected.empty:
        _view = dqn_pred_selected[dqn_pred_selected["evaluation_split"] == "view_build"]
        _test = dqn_pred_selected[dqn_pred_selected["evaluation_split"] == "test"]
        pred_view_aug = pd.concat([pred_view_aug, _view], ignore_index=True)
        pred_test_aug = pd.concat([pred_test_aug, _test], ignore_index=True)
        print(f"[DQN_pred] Selected {len(_test)} test predictions")

    # ── 6b. PPO_pred ─────────────────────────────────────────────────────────
    print("\n" + "─" * 90)
    print("=== Agent 2/4: PPO_pred (per-asset, prediction accuracy) ===")
    print("─" * 90)
    ppo_pred_selected, ppo_pred_diag = build_ppo_pred_selected_predictions(
        per_prediction=per_pred,
        candidate_models=model_names,
        train_split="view_build",
        infer_splits=("view_build", "test"),
        output_model_name="PPO_Pred",
        params=dict(hidden_dims=(128, 64), gamma=0.95, lam=0.95, lr=3e-4,
                    epochs=12, device=device_str, verbose=True),
        verbose=True,
    )
    ppo_pred_diag.to_csv(OUTPUT_DIR / f"{run_name}_ppo_pred_diagnostics.csv", index=False)
    if not ppo_pred_selected.empty:
        _view = ppo_pred_selected[ppo_pred_selected["evaluation_split"] == "view_build"]
        _test = ppo_pred_selected[ppo_pred_selected["evaluation_split"] == "test"]
        pred_view_aug = pd.concat([pred_view_aug, _view], ignore_index=True)
        pred_test_aug = pd.concat([pred_test_aug, _test], ignore_index=True)
        print(f"[PPO_pred] Selected {len(_test)} test predictions")

    # ── 6c. DQN_port ─────────────────────────────────────────────────────────
    print("\n" + "─" * 90)
    print("=== Agent 3/4: DQN_port (cross-sectional, Sharpe reward) ===")
    print("─" * 90)
    dqn_port = DQNPortSelector(
        model_names=model_names,
        config=DQNPortConfig(
            episodes=25, gamma=0.9, learning_rate=1e-3, batch_size=256,
            hidden_dims=(256, 256), target_update_every=200,
            epsilon_start=1.0, epsilon_end=0.05, epsilon_decay=0.997,
            replay_capacity=50_000, ema_alpha=0.1, reward_scale=10.0,
            horizon_days=HORIZON_DAYS, periods_per_year=PERIODS_PER_YEAR,
            seed=42, device=device_str,
        ),
    )
    dqn_port_log = dqn_port.fit(train_states, reward_panel)
    dqn_port_log.to_csv(OUTPUT_DIR / f"{run_name}_dqn_port_training_log.csv", index=False)

    dqn_port_actions_view = dqn_port.predict_actions(train_states)
    dqn_port_actions_test = dqn_port.predict_actions(test_states)
    dqn_port_actions_test.to_csv(OUTPUT_DIR / f"{run_name}_dqn_port_actions_test.csv", index=False)

    dqn_port_pred_view = build_selected_predictions(per_pred_view, dqn_port_actions_view, "DQN_Port")
    dqn_port_pred_test = build_selected_predictions(per_pred_test, dqn_port_actions_test, "DQN_Port")
    pred_view_aug = pd.concat([pred_view_aug, dqn_port_pred_view], ignore_index=True)
    pred_test_aug = pd.concat([pred_test_aug, dqn_port_pred_test], ignore_index=True)

    mix = dqn_port_actions_test["selected_model"].value_counts(normalize=True)
    print(f"\n[DQN_port] Model mix (test):\n{mix.to_string()}")

    # ── 6d. PPO_port ─────────────────────────────────────────────────────────
    print("\n" + "─" * 90)
    print("=== Agent 4/4: PPO_port (cross-sectional, Sharpe reward) ===")
    print("─" * 90)
    ppo_port = PPOPortSelector(
        model_names=model_names,
        config=PPOPortConfig(
            epochs=25, gamma=0.9, lam=0.95, learning_rate=3e-4,
            clip_eps=0.2, entropy_coef=0.01, value_coef=0.5,
            hidden_dims=(256, 256), mini_batch_size=256, ppo_update_epochs=4,
            ema_alpha=0.1, reward_scale=10.0,
            horizon_days=HORIZON_DAYS, periods_per_year=PERIODS_PER_YEAR,
            seed=42, device=device_str,
        ),
    )
    ppo_port_log = ppo_port.fit(train_states, reward_panel)
    ppo_port_log.to_csv(OUTPUT_DIR / f"{run_name}_ppo_port_training_log.csv", index=False)

    ppo_port_actions_view = ppo_port.predict_actions(train_states)
    ppo_port_actions_test = ppo_port.predict_actions(test_states)
    ppo_port_actions_test.to_csv(OUTPUT_DIR / f"{run_name}_ppo_port_actions_test.csv", index=False)

    ppo_port_pred_view = build_selected_predictions(per_pred_view, ppo_port_actions_view, "PPO_Port")
    ppo_port_pred_test = build_selected_predictions(per_pred_test, ppo_port_actions_test, "PPO_Port")
    pred_view_aug = pd.concat([pred_view_aug, ppo_port_pred_view], ignore_index=True)
    pred_test_aug = pd.concat([pred_test_aug, ppo_port_pred_test], ignore_index=True)

    mix = ppo_port_actions_test["selected_model"].value_counts(normalize=True)
    print(f"\n[PPO_port] Model mix (test):\n{mix.to_string()}")

    # ══════════════════════════════════════════════════════════════════════════
    # 7. Strategy definitions
    # ══════════════════════════════════════════════════════════════════════════
    rl_model_names = ["DQN_Pred", "PPO_Pred", "DQN_Port", "PPO_Port"]
    strategies = [
        StaticBLStrategyConfig(name="BL_NoView", mode="no_view", fallback_to_no_view=False),
    ]

    # 4 RL agents
    for rl_name in rl_model_names:
        strategies.append(StaticBLStrategyConfig(
            name=f"BL_{rl_name}", mode="absolute",
            absolute_model_name=rl_name, absolute_top_k=TOP_K_VIEWS,
            omega_method="uncertainty", annualize_q=False, annualize_omega=False,
            omega_scale=OMEGA_SCALE, horizon_days=HORIZON_DAYS,
            periods_per_year=PERIODS_PER_YEAR, fallback_to_no_view=True,
        ))

    # All individual ML/DL models
    for m in model_names:
        strategies.append(StaticBLStrategyConfig(
            name=f"BL_{m}", mode="absolute",
            absolute_model_name=m, absolute_top_k=TOP_K_VIEWS,
            omega_method="uncertainty", annualize_q=False, annualize_omega=False,
            omega_scale=OMEGA_SCALE, horizon_days=HORIZON_DAYS,
            periods_per_year=PERIODS_PER_YEAR, fallback_to_no_view=True,
        ))

    # ══════════════════════════════════════════════════════════════════════════
    # 8. Walk-forward BL portfolio evaluation
    # ══════════════════════════════════════════════════════════════════════════
    print(f"\n=== Walk-forward evaluation: {len(strategies)} strategies ===")
    result = run_static_bl_view_strategies(
        close_prices=close_prices,
        splits=splits,
        per_prediction_test=pred_test_aug,
        per_prediction_view=pred_view_aug,
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
        market_weights=market_weights,
        market_caps=market_caps,
        tau=TAU,
        risk_aversion=RISK_AVERSION,
        transaction_cost=0.0,
        charge_initial_cost=False,
        benchmark_strategy_name="BL_NoView",
        verbose_progress=True,
    )

    # ══════════════════════════════════════════════════════════════════════════
    # 9. Results
    # ══════════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 90)
    print("=== AGGREGATE PERFORMANCE SUMMARY ===")
    print("=" * 90)
    summary = result.aggregate_summary.round(6)
    print(summary.to_string())

    # Highlight RL agents vs benchmark
    print("\n=== RL agents vs BL_NoView ===")
    rl_rows = ["BL_NoView"] + [f"BL_{n}" for n in rl_model_names]
    rl_summary = summary.loc[summary.index.isin(rl_rows)]
    if not rl_summary.empty:
        cols = ["mean_sharpe_ratio", "mean_total_return", "mean_max_drawdown",
                "mean_annualized_volatility", "win_vs_benchmark_sharpe"]
        print(rl_summary[[c for c in cols if c in rl_summary.columns]].to_string())

    # Save
    result.aggregate_summary.reset_index().to_csv(
        OUTPUT_DIR / f"{run_name}_aggregate_summary.csv", index=False)
    result.fold_summary_long.to_csv(
        OUTPUT_DIR / f"{run_name}_fold_summary_long.csv", index=False)
    result.step_results.to_csv(
        OUTPUT_DIR / f"{run_name}_step_results.csv", index=False)
    mape_df.to_csv(OUTPUT_DIR / f"{run_name}_mape_by_model.csv", index=False)

    print("\n=== Output files ===")
    for f in sorted(OUTPUT_DIR.glob(f"{run_name}_*")):
        print(f"  {f}")


def main() -> None:
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 320)

    FORECAST_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    run_experiment(DATASET_PATH)
    print("\n=== All done ===")


if __name__ == "__main__":
    main()
