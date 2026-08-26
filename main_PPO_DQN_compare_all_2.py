from __future__ import annotations

"""
main_PPO_DQN_compare_all_2.py
==============================
Enhanced version of main_PPO_DQN_compare_all.py with:
  1. Improved forecasting features (from main_forecast.py)
     - Derived features (log-returns, volatility, MA ratios, cross-sectional norm)
     - Macro variables → change rates
     - Drop close / MarketCap_USD (ill-conditioning fix)
     - Uses target_5d (raw) for interpretability + tuned hyperparameters
  2. Hybrid Omega matrix
     - base_variance × model_quality (Rank IC) × prediction_strength × scale
     - Applied in both RL reward computation and final BL backtest
  3. Longer RL training
     - DQN_pred/PPO_pred: 12 → 30 episodes
     - DQN_port/PPO_port: 25 → 60 episodes

Agents compared:
  1. DQN_pred  — per-asset selection, reward = prediction accuracy
  2. DQN_port  — cross-sectional selection, reward = EMA-Sharpe (BL return)
  3. PPO_pred  — per-asset selection, reward = prediction accuracy
  4. PPO_port  — cross-sectional selection, reward = EMA-Sharpe (BL return)
"""

from pathlib import Path
import time

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from scipy.stats import spearmanr

# ── Data / forecasting infrastructure ────────────────────────────────────────
from src.data.csv_adapter import load_final_clean_result, build_asset_panel
from src.data.splitters import build_nested_walk_forward_splits, describe_split_coverage
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
OMEGA_SCALE            = 0.1    # ↓ v2 (was 1.0): weaken hybrid Omega

TARGET_COL             = "target_5d"   # raw 5d return (no rf adjustment)

# Hybrid Omega hyperparams — weakened for v2
HYBRID_QUALITY_FLOOR    = 0.005
HYBRID_STRENGTH_FLOOR   = 1e-3
HYBRID_QUALITY_EXPONENT = 0.25   # ↓ v2 (was 0.5): gentler quality penalty
HYBRID_STRENGTH_EXPONENT = 0.25  # ↓ v2 (was 0.5): gentler strength penalty
HYBRID_MIN_VAR          = 1e-8
HYBRID_MAX_VAR          = 1.0

# Reuse improved forecasts from main_forecast.py if available
FORECAST_DIR           = Path("outputs/forecast_improved")
FORECAST_RUN_NAME      = "forecast_v2"

RUN_NAME               = "rl_compare_all_v2"
OUTPUT_DIR             = Path("outputs/rl_compare_all_v2")


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
# Feature engineering (from main_forecast.py)
# =============================================================================
def add_derived_features(panel: pd.DataFrame) -> pd.DataFrame:
    """
    Add per-asset derived features + cross-sectional normalisation.
    No look-ahead — all features are backward-looking.
    """
    out = panel.sort_values(["asset", "date"]).copy()

    # 1. Macro level → change rates
    macro_cols = ["VIX", "MOVE", "HY_OAS", "Spread_10Y2Y", "Dollar_Index"]
    macro_by_date = out.groupby("date")[
        [c for c in macro_cols if c in out.columns]
    ].first().sort_index()

    for col in macro_cols:
        if col not in macro_by_date.columns:
            continue
        series = macro_by_date[col].astype(float)
        chg_1d = series.pct_change(1).rename(f"{col}_chg1d")
        chg_5d = series.pct_change(5).rename(f"{col}_chg5d")
        ma20 = series.rolling(20, min_periods=10).mean()
        ratio = (series / ma20).rename(f"{col}_ma20ratio")
        derived = pd.concat([chg_1d, chg_5d, ratio], axis=1).reset_index()
        out = out.merge(derived, on="date", how="left")

    out = out.drop(columns=[c for c in macro_cols if c in out.columns], errors="ignore")

    # 2. Per-asset features
    derived_parts = []
    for _, g in out.groupby("asset"):
        g = g.sort_values("date").copy()
        px = g["close"].astype(float)
        log_ret = np.log(px / px.shift(1))

        g["log_ret_1d"]  = log_ret
        g["log_ret_5d"]  = np.log(px / px.shift(5))
        g["log_ret_10d"] = np.log(px / px.shift(10))
        g["log_ret_20d"] = np.log(px / px.shift(20))
        g["rvol_5d"]  = log_ret.rolling(5,  min_periods=3).std()
        g["rvol_20d"] = log_ret.rolling(20, min_periods=10).std()
        g["vol_ratio_5_20"] = g["rvol_5d"] / g["rvol_20d"].replace(0, np.nan)

        ma5  = px.rolling(5,  min_periods=3).mean()
        ma20 = px.rolling(20, min_periods=10).mean()
        ma60 = px.rolling(60, min_periods=30).mean()
        g["px_ma5_ratio"]   = px / ma5.replace(0, np.nan)
        g["px_ma20_ratio"]  = px / ma20.replace(0, np.nan)
        g["px_ma60_ratio"]  = px / ma60.replace(0, np.nan)
        g["ma5_ma20_ratio"] = ma5 / ma20.replace(0, np.nan)
        g["ret_skew_20d"]   = log_ret.rolling(20, min_periods=10).skew()

        if "RSI_14" in g.columns:
            g["rsi_delta_5d"] = g["RSI_14"].astype(float).diff(5)
        if "ATR_14_pct" in g.columns:
            g["atr_regime"] = g["ATR_14_pct"].astype(float).rolling(20, min_periods=10).mean()
        if "DPO_20" in g.columns:
            g["DPO_20"] = g["DPO_20"].astype(float) / px.replace(0, np.nan)
        if "MarketCap_Weight" in g.columns:
            g["mcap_wt_chg5d"] = g["MarketCap_Weight"].astype(float).pct_change(5)

        derived_parts.append(g)
    out = pd.concat(derived_parts, ignore_index=True)

    # 3. Cross-sectional normalisation
    cs_cols = [
        "log_ret_1d", "log_ret_5d", "log_ret_10d", "log_ret_20d",
        "rvol_5d", "rvol_20d", "vol_ratio_5_20",
        "px_ma5_ratio", "px_ma20_ratio", "px_ma60_ratio", "ma5_ma20_ratio",
        "ret_skew_20d", "rsi_delta_5d",
        "RSI_14", "StochRSI_14", "ROC_10", "TSI_25_13", "DPO_20",
        "ATR_14_pct", "atr_regime",
    ]
    cs_cols = [c for c in cs_cols if c in out.columns]
    print(f"[INFO] Cross-sectional normalisation on {len(cs_cols)} features…")
    gb = out.groupby("date", sort=False)
    for col in cs_cols:
        out[f"{col}_csrank"] = gb[col].transform(lambda s: s.rank(pct=True))
        out[f"{col}_cszscore"] = gb[col].transform(
            lambda s: (s - s.mean()) / (s.std(ddof=0) + 1e-10)
        )

    out = out.replace([np.inf, -np.inf], np.nan)
    return out.sort_values(["asset", "date"]).reset_index(drop=True)


# =============================================================================
# Tuned forecaster registry (loads from main_forecast.py's best_params.csv)
# =============================================================================
def _load_best_params(path: Path) -> dict:
    if not path.exists():
        print(f"[WARN] Best params not found at {path}. Using sensible defaults.")
        return {}
    df = pd.read_csv(path)
    out = {}
    for model, g in df.groupby("model"):
        out[model] = {}
        for _, row in g.iterrows():
            v = row["value"]
            if isinstance(v, float) and v.is_integer():
                v = int(v)
            out[model][row["param"]] = v
    return out


def build_forecasters(best_params: dict) -> dict[str, object]:
    kw = dict(target_column=TARGET_COL, horizon_days=HORIZON_DAYS,
              periods_per_year=PERIODS_PER_YEAR)
    p = best_params

    xgb  = p.get("XGBoost", {})
    lgb  = p.get("LightGBM", {})
    cat  = p.get("CatBoost", {})
    rf   = p.get("RandomForest", {})
    lstm = p.get("LSTM", {})
    tf   = p.get("Transformer", {})
    ptst = p.get("PatchTST", {})
    cnn  = p.get("CNN1D", {})
    mlp  = p.get("MLP", {})
    hyb  = p.get("HybridLSTMTF", {})

    def _dl_kw(pr):
        return dict(
            epochs=pr.get("epochs", 30),
            batch_size=pr.get("batch_size", 512),
            verbose=False, **kw,
        )

    # CNN channels
    cnn_n = cnn.get("n_channels", 3)
    if "ch_0" in cnn:
        cnn_channels = tuple(cnn.get(f"ch_{i}", 64) for i in range(cnn_n))
    else:
        cnn_channels = (64, 128, 64)

    # MLP dims
    mlp_n = mlp.get("n_layers", 3)
    mlp_dims = tuple(mlp.get(f"dim_{i}", 256) for i in range(mlp_n))

    return {
        "Ridge":          LinearForecaster(linear_type="ridge", alpha=300.0, **kw),
        "Lasso":          LinearForecaster(linear_type="lasso", alpha=0.01, **kw),
        "DecisionTree":   DecisionTreeForecaster(max_depth=8, min_samples_leaf=50, **kw),
        "RandomForest":   RandomForestForecaster(
                              n_estimators=rf.get("n_estimators", 600),
                              max_depth=rf.get("max_depth", 10),
                              min_samples_leaf=rf.get("min_samples_leaf", 20), **kw),
        "XGBoost":        XGBoostForecaster(
                              n_estimators=xgb.get("n_estimators", 500),
                              learning_rate=xgb.get("lr", 0.01),
                              max_depth=xgb.get("max_depth", 5), **kw),
        "LightGBM":       LightGBMForecaster(
                              n_estimators=lgb.get("n_estimators", 600),
                              learning_rate=lgb.get("lr", 0.01),
                              num_leaves=lgb.get("num_leaves", 63), **kw),
        "CatBoost":       CatBoostForecaster(
                              iterations=cat.get("iterations", 600),
                              learning_rate=cat.get("lr", 0.01),
                              depth=cat.get("depth", 7), **kw),
        "CNN1D":          CNN1DForecaster(
                              lookback=cnn.get("lookback", 60),
                              channels=cnn_channels,
                              kernel_size=cnn.get("kernel_size", 5),
                              lr=cnn.get("lr", 5e-4),
                              dropout=cnn.get("dropout", 0.15),
                              **_dl_kw(cnn)),
        "MLP":            MLPForecaster(
                              lookback=mlp.get("lookback", 60),
                              hidden_dims=mlp_dims,
                              lr=mlp.get("lr", 5e-4),
                              dropout=mlp.get("dropout", 0.15),
                              **_dl_kw(mlp)),
        "LSTM":           LSTMForecaster(
                              lookback=lstm.get("lookback", 60),
                              hidden_dim=lstm.get("hidden_dim", 128),
                              num_layers=lstm.get("num_layers", 3),
                              lr=lstm.get("lr", 5e-4),
                              dropout=lstm.get("dropout", 0.15),
                              **_dl_kw(lstm)),
        "Transformer":    TransformerForecaster(
                              lookback=tf.get("lookback", 120),
                              d_model=tf.get("d_model", 128),
                              n_heads=tf.get("n_heads", 8),
                              num_layers=tf.get("num_layers", 3),
                              lr=tf.get("lr", 3e-4),
                              dropout=tf.get("dropout", 0.15),
                              **_dl_kw(tf)),
        "PatchTST":       PatchTSTForecaster(
                              lookback=ptst.get("lookback", 120),
                              patch_len=ptst.get("patch_len", 20),
                              stride=ptst.get("stride", 10),
                              lr=ptst.get("lr", 3e-4),
                              dropout=ptst.get("dropout", 0.15),
                              **_dl_kw(ptst)),
        "HybridLSTMTransformer": HybridLSTMTransformerForecaster(
                              lookback=hyb.get("lookback", lstm.get("lookback", 60)),
                              hidden_dim=hyb.get("hidden_dim", 128),
                              lstm_layers=hyb.get("lstm_layers", 2),
                              d_model=hyb.get("d_model", 128),
                              n_heads=hyb.get("n_heads", 8),
                              transformer_layers=hyb.get("tf_layers", 2),
                              lr=hyb.get("lr", 3e-4),
                              dropout=hyb.get("dropout", 0.15),
                              **_dl_kw(hyb)),
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
# State builder for _port agents
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
# Hybrid Omega helpers
# =============================================================================
def compute_model_quality_scores(per_pred_view: pd.DataFrame) -> dict[str, float]:
    """
    Compute Rank IC for each model on view_build set.
    Used as model quality factor in hybrid Omega.
    """
    scores = {}
    for model, g in per_pred_view.groupby("model"):
        yt = pd.to_numeric(g["y_true"], errors="coerce")
        yp = pd.to_numeric(g["y_pred"], errors="coerce")
        valid = yt.notna() & yp.notna()
        if valid.sum() < 10:
            scores[model] = HYBRID_QUALITY_FLOOR
            continue
        corr, _ = spearmanr(yt[valid], yp[valid])
        scores[model] = float(corr) if np.isfinite(corr) else HYBRID_QUALITY_FLOOR
    return scores


def hybrid_unc_adjustment(
    base_unc: float,
    y_pred: float,
    model_quality: float,
) -> float:
    """
    Pre-adjust uncertainty column so omega_method='uncertainty' produces hybrid Omega.

    Target: omega_var = unc² × quality_mult × strength_mult × scale
    Solve: adjusted_unc = unc × sqrt(quality_mult × strength_mult)
    """
    ic = max(float(model_quality), HYBRID_QUALITY_FLOOR)
    q_mult = (1.0 / ic) ** HYBRID_QUALITY_EXPONENT
    abs_q = max(abs(float(y_pred)), HYBRID_STRENGTH_FLOOR)
    s_mult = (1.0 / abs_q) ** HYBRID_STRENGTH_EXPONENT
    return float(base_unc) * float(np.sqrt(q_mult * s_mult))


def apply_hybrid_omega_to_predictions(
    per_pred: pd.DataFrame,
    model_quality_scores: dict[str, float],
) -> pd.DataFrame:
    """
    Adjust per_pred's 'uncertainty' column so that the existing omega_method='uncertainty'
    path produces a hybrid Omega matrix when used by run_static_bl_view_strategies.

    - Non-hybrid columns preserved
    - If 'uncertainty' missing or NaN, use |y_pred| * 0.5 as base
    """
    out = per_pred.copy()
    if "uncertainty" not in out.columns:
        out["uncertainty"] = out["y_pred"].abs() * 0.5
    else:
        out["uncertainty"] = out["uncertainty"].fillna(out["y_pred"].abs() * 0.5)

    # Apply adjustment per row
    def _adjust(row):
        q = model_quality_scores.get(row["model"], HYBRID_QUALITY_FLOOR)
        return hybrid_unc_adjustment(row["uncertainty"], row["y_pred"], q)

    out["uncertainty_raw"] = out["uncertainty"].values
    out["uncertainty"] = out.apply(_adjust, axis=1).clip(
        lower=np.sqrt(HYBRID_MIN_VAR), upper=np.sqrt(HYBRID_MAX_VAR),
    )
    return out


# =============================================================================
# BL reward panel with hybrid Omega
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


def _build_bl_views_hybrid(
    per_pred, fold_id, decision_date, model_name, universe,
    model_quality_scores, top_k=TOP_K_VIEWS, omega_scale=OMEGA_SCALE,
):
    """Build BL P, Q, Omega using hybrid Omega (already per-row adjusted)."""
    snap = per_pred[
        (per_pred["fold_id"] == fold_id)
        & (pd.to_datetime(per_pred["date"]) == pd.Timestamp(decision_date))
        & (per_pred["model"] == model_name)
    ].copy()
    if snap.empty:
        raise ValueError(f"No predictions: fold={fold_id} date={decision_date} model={model_name}")

    # Use uncertainty_raw (original) and apply hybrid formula here
    if "uncertainty_raw" in snap.columns:
        snap["_unc"] = snap["uncertainty_raw"].astype(float).fillna(snap["y_pred"].abs() * 0.5)
    elif "uncertainty" in snap.columns and snap["uncertainty"].notna().any():
        snap["_unc"] = snap["uncertainty"].astype(float).fillna(snap["y_pred"].abs() * 0.5)
    else:
        snap["_unc"] = snap["y_pred"].abs() * 0.5

    snap = snap.assign(abs_pred=snap["y_pred"].abs()).nlargest(min(top_k, len(snap)), "abs_pred")

    ic = max(model_quality_scores.get(model_name, HYBRID_QUALITY_FLOOR), HYBRID_QUALITY_FLOOR)
    q_mult = (1.0 / ic) ** HYBRID_QUALITY_EXPONENT

    P, Q, omega_diag = [], [], []
    for _, row in snap.iterrows():
        asset = row["asset"]
        if asset not in universe:
            continue
        vec = np.zeros(len(universe), dtype=float)
        vec[universe.index(asset)] = 1.0
        P.append(vec)
        Q.append(float(row["y_pred"]))
        base_unc = max(float(row["_unc"]), 1e-6)
        abs_q = max(abs(float(row["y_pred"])), HYBRID_STRENGTH_FLOOR)
        s_mult = (1.0 / abs_q) ** HYBRID_STRENGTH_EXPONENT
        omega_var = (base_unc ** 2) * q_mult * s_mult * omega_scale
        omega_var = float(np.clip(omega_var, HYBRID_MIN_VAR, HYBRID_MAX_VAR))
        omega_diag.append(omega_var)

    if not P:
        raise ValueError("No valid BL views.")
    return (
        np.array(P, dtype=float),
        np.array(Q, dtype=float),
        np.diag(omega_diag),
    )


def _ensure_1d(x, index, name):
    if isinstance(x, pd.Series):
        return x.copy().reindex(index).rename(name).astype(float)
    if isinstance(x, pd.DataFrame):
        x = x.iloc[:, 0] if x.shape[1] == 1 else x.iloc[0, :]
        return x.reindex(index).rename(name).astype(float)
    return pd.Series(np.asarray(x).reshape(-1), index=index, name=name, dtype=float)


def _compute_bl_portfolio_return(
    close_prices, per_pred, fold_id, decision_date, forecast_end_date,
    model_name, market_caps, market_weights, model_quality_scores,
):
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
        P, Q, Omega = _build_bl_views_hybrid(
            per_pred, fold_id, decision_date, model_name, universe,
            model_quality_scores,
        )
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


def build_bl_reward_panel(
    close_prices, per_pred_view, splits, model_names,
    market_caps, market_weights, model_quality_scores,
):
    all_rows = []
    fold_logger = ProgressLogger(len(splits), "BL reward (folds)", 1)
    for _, sr in splits.iterrows():
        fold_id = int(sr["fold_id"])
        schedule = build_rebalance_schedule(
            close_prices, pd.Timestamp(sr["view_start"]), pd.Timestamp(sr["view_end"]),
        )
        n_jobs = max(len(schedule) * len(model_names), 1)
        step_logger = ProgressLogger(n_jobs, f"fold={fold_id}", max(n_jobs // 10, 1))

        for row in schedule.itertuples(index=False):
            d1, d2 = pd.Timestamp(row.decision_date), pd.Timestamp(row.forecast_end_date)
            out_row = {"fold_id": fold_id, "date": d1}
            for m in model_names:
                out_row[f"reward_blret__{m}"] = _compute_bl_portfolio_return(
                    close_prices, per_pred_view, fold_id, d1, d2, m,
                    market_caps, market_weights, model_quality_scores,
                )
                step_logger.update(1, extra=f"{d1.date()} {m}")
            all_rows.append(out_row)
        fold_logger.update(1, extra=f"fold={fold_id} steps={len(schedule)}")
    return pd.DataFrame(all_rows).sort_values(["fold_id", "date"]).reset_index(drop=True)


# =============================================================================
# Cache loaders
# =============================================================================
def load_or_run_forecasts(dataset_tag, panel, splits, forecasters):
    """Try loading cached forecast from main_forecast.py; else run from scratch."""
    cache_parq = FORECAST_DIR / f"{FORECAST_RUN_NAME}_per_prediction.parquet"
    cache_csv  = FORECAST_DIR / f"{FORECAST_RUN_NAME}_per_prediction.csv"

    if cache_parq.exists():
        print(f"[INFO] Loading forecast cache: {cache_parq}")
        df = pd.read_parquet(cache_parq)
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"])
        return df
    if cache_csv.exists():
        print(f"[INFO] Loading forecast cache: {cache_csv}")
        return pd.read_csv(cache_csv, parse_dates=["date"])

    print("[INFO] No forecast cache — running all forecasters (slow)…")
    r_view = run_forecast_baselines_on_splits(
        panel_df=panel, splits=splits, forecasters=forecasters,
        target_col=TARGET_COL, evaluation_split="view_build",
        exclude_target_like=True, date_col="date", verbose_progress=True)
    r_test = run_forecast_baselines_on_splits(
        panel_df=panel, splits=splits, forecasters=forecasters,
        target_col=TARGET_COL, evaluation_split="test",
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
        output_dir=FORECAST_DIR, run_name=FORECAST_RUN_NAME,
        save_csv=True, save_parquet=True)
    return per_pred


def load_or_build_reward_panel(
    run_name, close_prices, per_pred_view, splits, model_names,
    market_caps, market_weights, model_quality_scores,
):
    path = OUTPUT_DIR / f"{run_name}_reward_panel.csv"
    if path.exists():
        print(f"[INFO] Loading reward panel: {path}")
        return pd.read_csv(path, parse_dates=["date"])
    print("[INFO] Building BL reward panel with hybrid Omega (slow step)…")
    panel = build_bl_reward_panel(
        close_prices, per_pred_view, splits, model_names,
        market_caps, market_weights, model_quality_scores,
    )
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


def calc_rank_ic(y_true, y_pred):
    corr, _ = spearmanr(y_true, y_pred, nan_policy="omit")
    return float(corr) if np.isfinite(corr) else np.nan


def calc_wmape(y_true, y_pred):
    yt, yp = np.asarray(y_true, float), np.asarray(y_pred, float)
    d = np.sum(np.abs(yt))
    return float("nan") if d < 1e-12 else float(np.sum(np.abs(yt - yp)) / d * 100)


# =============================================================================
# Main experiment
# =============================================================================
def run_experiment(csv_path: str) -> None:
    dataset_tag = Path(csv_path).stem.lower()
    run_name = f"{dataset_tag}_{RUN_NAME}"

    print("\n" + "=" * 90)
    print(f"=== 4-Agent RL Comparison (v2 with Hybrid Omega): {csv_path} ===")
    print("=" * 90)

    device_str = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] Device: {device_str}")
    print(f"[INFO] Target: {TARGET_COL} (raw return, no rf adjustment)")
    print(f"[INFO] Omega method: hybrid (base × quality^{HYBRID_QUALITY_EXPONENT} × strength^{HYBRID_STRENGTH_EXPONENT})")

    # ── 1. Data + derived features ──────────────────────────────────────────
    raw_df = load_final_clean_result(csv_path, date_col="Date")
    panel = build_asset_panel(
        raw_df, target_prefix="Return_5d_", target_name="target_5d",
        include_shared=True, include_close=True, dropna_target=True)
    print(f"[INFO] Raw panel: {len(panel)} rows, {panel['asset'].nunique()} assets")

    print("\n[INFO] Adding derived features…")
    panel = add_derived_features(panel)

    # Drop ill-conditioning columns (after using them to build derived features)
    panel = panel.drop(columns=[c for c in ["close", "MarketCap_USD"] if c in panel.columns], errors="ignore")

    feature_cols = [c for c in panel.columns
                    if c not in ["date", "asset", "target_5d"]
                    and "target" not in c.lower() and "return" not in c.lower()
                    and "future" not in c.lower()]
    print(f"[INFO] Total features: {len(feature_cols)}")

    close_prices = extract_close_price_table_from_wide_df(raw_df, date_col="Date")
    assets = close_prices.columns.tolist()
    print(f"Universe: {len(assets)} assets")

    dataset_caps, dataset_weights = extract_market_caps_and_weights_from_wide_df(raw_df, assets)
    market_caps, market_weights, wt_src = resolve_reference_weights(dataset_caps, dataset_weights, assets)
    print(f"[INFO] Market weight source: {wt_src}")

    # ── 2. Splits ───────────────────────────────────────────────────────────
    splits = build_nested_walk_forward_splits(
        panel, model_train_days=MODEL_TRAIN_DAYS, view_build_days=VIEW_BUILD_DAYS,
        test_days=TEST_DAYS, step_days=STEP_DAYS, date_col="date")
    print(f"\nSplits: {len(splits)} folds")
    print(describe_split_coverage(splits).to_string(index=False))

    # ── 3. Forecasting (use tuned params from main_forecast.py) ────────────
    best_params_path = FORECAST_DIR / f"{FORECAST_RUN_NAME}_best_params.csv"
    best_params = _load_best_params(best_params_path)
    if best_params:
        print(f"[INFO] Loaded tuned params for {len(best_params)} models from {best_params_path}")
    forecasters = build_forecasters(best_params)
    model_names = list(forecasters.keys())
    print(f"\nForecasting models ({len(model_names)}): {model_names}")

    per_pred = load_or_run_forecasts(dataset_tag, panel, splits, forecasters)
    per_pred_view = per_pred[per_pred["evaluation_split"] == "view_build"].copy()
    per_pred_test = per_pred[per_pred["evaluation_split"] == "test"].copy()

    # ── 4. Model quality (Rank IC) for hybrid Omega ─────────────────────────
    print("\n[INFO] Computing model quality (Rank IC on view_build)…")
    model_quality_scores = compute_model_quality_scores(per_pred_view)
    quality_df = pd.DataFrame([
        {"model": m, "rank_ic": q} for m, q in model_quality_scores.items()
    ]).sort_values("rank_ic", ascending=False)
    print(quality_df.round(5).to_string(index=False))
    quality_df.to_csv(OUTPUT_DIR / f"{run_name}_model_quality.csv", index=False)

    # Quick per-model metrics (test)
    print("\n=== Per-model test metrics ===")
    test_metrics = []
    for m, g in per_pred_test.groupby("model"):
        yt = pd.to_numeric(g["y_true"], errors="coerce")
        yp = pd.to_numeric(g["y_pred"], errors="coerce")
        v = yt.notna() & yp.notna()
        test_metrics.append({
            "model": m,
            "wmape_pct": calc_wmape(yt[v], yp[v]),
            "rank_ic": calc_rank_ic(yt[v], yp[v]),
            "n": int(v.sum()),
        })
    print(pd.DataFrame(test_metrics).round(5).to_string(index=False))

    # ── 5. State tables (_port agents) ──────────────────────────────────────
    print("\n[INFO] Building state tables for _port agents…")
    raw_train_st = build_state_table(per_pred_view, model_names)
    raw_test_st = build_state_table(per_pred_test, model_names)
    train_states, test_states, _ = normalise_states(raw_train_st, raw_test_st)
    print(f"[INFO] State dim={len(train_states.iloc[0]['state'])} | "
          f"train={len(train_states)} test={len(test_states)}")

    # ── 6. BL reward panel with hybrid Omega ────────────────────────────────
    reward_panel = load_or_build_reward_panel(
        run_name, close_prices, per_pred_view, splits,
        model_names, market_caps, market_weights, model_quality_scores,
    )
    print(f"[INFO] Reward panel: {len(reward_panel)} rows")

    # ══════════════════════════════════════════════════════════════════════════
    # 7. Train all 4 RL agents (with more episodes than v1)
    # ══════════════════════════════════════════════════════════════════════════
    pred_view_aug = per_pred_view.copy()
    pred_test_aug = per_pred_test.copy()

    # ── 7a. DQN_pred ────────────────────────────────────────────────────────
    print("\n" + "─" * 90)
    print("=== Agent 1/4: DQN_pred (per-asset, 50 episodes, bigger net) ===")
    print("─" * 90)
    dqn_pred_selected, dqn_pred_diag = build_dqn_pred_selected_predictions(
        per_prediction=per_pred, candidate_models=model_names,
        train_split="view_build", infer_splits=("view_build", "test"),
        output_model_name="DQN_Pred",
        params=dict(
            hidden_dims=(512, 256, 128),       # ↑ deeper/wider
            gamma=0.9,                         # ↓ shorter horizon (less noise)
            lr=3e-4,                           # ↓ more stable
            episodes=50,                       # ↑ more training
            batch_size=512,                    # ↑ bigger batches
            replay_capacity=200_000,           # ↑ more memory
            min_replay_size=2048,              # ↑ warm-up
            target_update_interval=500,        # ↑ more stable targets
            epsilon_start=1.0, epsilon_end=0.03,
            epsilon_decay=0.98,                # slower decay → more exploration
            device=device_str, verbose=True,
        ),
        verbose=True,
    )
    dqn_pred_diag.to_csv(OUTPUT_DIR / f"{run_name}_dqn_pred_diagnostics.csv", index=False)
    if not dqn_pred_selected.empty:
        _view = dqn_pred_selected[dqn_pred_selected["evaluation_split"] == "view_build"]
        _test = dqn_pred_selected[dqn_pred_selected["evaluation_split"] == "test"]
        pred_view_aug = pd.concat([pred_view_aug, _view], ignore_index=True)
        pred_test_aug = pd.concat([pred_test_aug, _test], ignore_index=True)
        print(f"[DQN_pred] Selected {len(_test)} test predictions")

    # ── 7b. PPO_pred ────────────────────────────────────────────────────────
    print("\n" + "─" * 90)
    print("=== Agent 2/4: PPO_pred (per-asset, 50 epochs, bigger net) ===")
    print("─" * 90)
    ppo_pred_selected, ppo_pred_diag = build_ppo_pred_selected_predictions(
        per_prediction=per_pred, candidate_models=model_names,
        train_split="view_build", infer_splits=("view_build", "test"),
        output_model_name="PPO_Pred",
        params=dict(
            hidden_dims=(512, 256, 128),       # ↑ deeper/wider
            gamma=0.9, lam=0.95,
            lr=2e-4,                           # ↓ stable
            clip_eps=0.15,                     # ↓ tighter clipping
            entropy_coef=0.02,                 # ↑ more exploration
            value_coef=0.5,
            max_grad_norm=0.5,
            epochs=50,                         # ↑ more training
            mini_batch_size=512,               # ↑ bigger batches
            ppo_update_epochs=10,              # ↑ more gradient steps per rollout
            device=device_str, verbose=True,
        ),
        verbose=True,
    )
    ppo_pred_diag.to_csv(OUTPUT_DIR / f"{run_name}_ppo_pred_diagnostics.csv", index=False)
    if not ppo_pred_selected.empty:
        _view = ppo_pred_selected[ppo_pred_selected["evaluation_split"] == "view_build"]
        _test = ppo_pred_selected[ppo_pred_selected["evaluation_split"] == "test"]
        pred_view_aug = pd.concat([pred_view_aug, _view], ignore_index=True)
        pred_test_aug = pd.concat([pred_test_aug, _test], ignore_index=True)
        print(f"[PPO_pred] Selected {len(_test)} test predictions")

    # ── 7c. DQN_port ────────────────────────────────────────────────────────
    print("\n" + "─" * 90)
    print("=== Agent 3/4: DQN_port (DSR reward, risk-adjusted) ===")
    print("─" * 90)
    dqn_port = DQNPortSelector(
        model_names=model_names,
        config=DQNPortConfig(
            episodes=80,
            gamma=0.9,
            learning_rate=3e-4,
            batch_size=512,
            hidden_dims=(256, 256, 128),
            target_update_every=500,
            epsilon_start=1.0, epsilon_end=0.03,
            epsilon_decay=0.99,
            replay_capacity=200_000,
            # NEW reward design
            reward_type="dsr",                 # Differential Sharpe (bounded, stable)
            dsr_eta=0.04,
            reward_scale=1.0,                  # ↓ massively (DSR is already bounded)
            reward_normalise=True,             # running z-score
            risk_penalty=0.3,                  # penalise volatile picks
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
    print(f"\n[DQN_port] Model mix (test):\n{dqn_port_actions_test['selected_model'].value_counts(normalize=True).to_string()}")

    # ── 7d. PPO_port ────────────────────────────────────────────────────────
    print("\n" + "─" * 90)
    print("=== Agent 4/4: PPO_port (DSR reward, risk-adjusted) ===")
    print("─" * 90)
    ppo_port = PPOPortSelector(
        model_names=model_names,
        config=PPOPortConfig(
            epochs=80,
            gamma=0.9,
            lam=0.95,
            learning_rate=3e-4,
            clip_eps=0.2,
            entropy_coef=0.02,
            value_coef=0.5,
            max_grad_norm=0.5,
            hidden_dims=(256, 256, 128),
            mini_batch_size=512,
            ppo_update_epochs=8,
            # NEW reward design
            reward_type="dsr",
            dsr_eta=0.04,
            reward_scale=1.0,
            reward_normalise=True,
            risk_penalty=0.3,
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
    print(f"\n[PPO_port] Model mix (test):\n{ppo_port_actions_test['selected_model'].value_counts(normalize=True).to_string()}")

    # ══════════════════════════════════════════════════════════════════════════
    # 8. Apply hybrid Omega to predictions (for BL backtest)
    # ══════════════════════════════════════════════════════════════════════════
    print("\n[INFO] Applying hybrid Omega adjustment to predictions…")
    # Compute quality for the RL-selected models too (mirror from source models)
    # For each RL-selected row, use the quality of its source model
    def _reassign_quality_for_rl(df):
        """RL selected rows have model=DQN_Pred/PPO_Pred/etc. Use source_model's quality."""
        out = df.copy()
        if "source_model" in out.columns:
            source_quality = out["source_model"].map(model_quality_scores)
            # model_quality_scores for RL models = best among source models they pick
            rl_quality = source_quality.fillna(HYBRID_QUALITY_FLOOR)
            # Create model_quality per-row
            out["_row_quality"] = rl_quality
        else:
            out["_row_quality"] = out["model"].map(
                lambda m: model_quality_scores.get(m, HYBRID_QUALITY_FLOOR)
            )
        return out

    # Augment quality dict for RL-named models using average of sources
    rl_model_names = ["DQN_Pred", "PPO_Pred", "DQN_Port", "PPO_Port"]
    for rl_name in rl_model_names:
        if rl_name in pred_test_aug["model"].unique():
            # Take mean quality of source models picked
            sel = pred_test_aug[pred_test_aug["model"] == rl_name]
            if "source_model" in sel.columns:
                src_qual = sel["source_model"].map(model_quality_scores).dropna()
                if not src_qual.empty:
                    model_quality_scores[rl_name] = float(src_qual.mean())
                else:
                    model_quality_scores[rl_name] = HYBRID_QUALITY_FLOOR
            else:
                model_quality_scores[rl_name] = HYBRID_QUALITY_FLOOR

    pred_view_aug = apply_hybrid_omega_to_predictions(pred_view_aug, model_quality_scores)
    pred_test_aug = apply_hybrid_omega_to_predictions(pred_test_aug, model_quality_scores)
    print(f"[INFO] Hybrid Omega: uncertainty column pre-adjusted for {len(model_quality_scores)} models")

    # ══════════════════════════════════════════════════════════════════════════
    # 9. Strategy definitions
    # ══════════════════════════════════════════════════════════════════════════
    strategies = [
        StaticBLStrategyConfig(name="BL_NoView", mode="no_view", fallback_to_no_view=False),
    ]
    for rl_name in rl_model_names:
        strategies.append(StaticBLStrategyConfig(
            name=f"BL_{rl_name}", mode="absolute",
            absolute_model_name=rl_name, absolute_top_k=TOP_K_VIEWS,
            omega_method="uncertainty",    # uses hybrid-adjusted uncertainty
            annualize_q=False, annualize_omega=False,
            omega_scale=OMEGA_SCALE, horizon_days=HORIZON_DAYS,
            periods_per_year=PERIODS_PER_YEAR, fallback_to_no_view=True,
        ))
    for m in model_names:
        strategies.append(StaticBLStrategyConfig(
            name=f"BL_{m}", mode="absolute",
            absolute_model_name=m, absolute_top_k=TOP_K_VIEWS,
            omega_method="uncertainty",    # uses hybrid-adjusted uncertainty
            annualize_q=False, annualize_omega=False,
            omega_scale=OMEGA_SCALE, horizon_days=HORIZON_DAYS,
            periods_per_year=PERIODS_PER_YEAR, fallback_to_no_view=True,
        ))

    # ══════════════════════════════════════════════════════════════════════════
    # 10. Walk-forward BL portfolio evaluation
    # ══════════════════════════════════════════════════════════════════════════
    print(f"\n=== Walk-forward evaluation: {len(strategies)} strategies ===")
    result = run_static_bl_view_strategies(
        close_prices=close_prices, splits=splits,
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
        market_weights=market_weights, market_caps=market_caps,
        tau=TAU, risk_aversion=RISK_AVERSION,
        transaction_cost=0.0, charge_initial_cost=False,
        benchmark_strategy_name="BL_NoView",
        verbose_progress=True,
    )

    # ══════════════════════════════════════════════════════════════════════════
    # 11. Results + Prediction↔Performance correlation
    # ══════════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 90)
    print("=== AGGREGATE PERFORMANCE SUMMARY ===")
    print("=" * 90)
    summary = result.aggregate_summary.round(6)
    print(summary.to_string())

    print("\n=== RL agents vs BL_NoView ===")
    rl_rows = ["BL_NoView"] + [f"BL_{n}" for n in rl_model_names]
    rl_summary = summary.loc[summary.index.isin(rl_rows)]
    if not rl_summary.empty:
        cols = ["mean_sharpe_ratio", "mean_total_return", "mean_max_drawdown",
                "mean_annualized_volatility", "win_vs_benchmark_sharpe"]
        print(rl_summary[[c for c in cols if c in rl_summary.columns]].to_string())

    # Prediction ↔ Portfolio correlation (for "better prediction → better investment")
    print("\n=== Prediction quality ↔ Portfolio performance correlation ===")
    corr_rows = []
    for m in model_names:
        strat_name = f"BL_{m}"
        if strat_name not in summary.index:
            continue
        corr_rows.append({
            "model": m,
            "rank_ic": model_quality_scores.get(m, np.nan),
            "sharpe": summary.loc[strat_name, "mean_sharpe_ratio"] if "mean_sharpe_ratio" in summary.columns else np.nan,
            "total_return": summary.loc[strat_name, "mean_total_return"] if "mean_total_return" in summary.columns else np.nan,
        })
    corr_df = pd.DataFrame(corr_rows)
    if not corr_df.empty and corr_df["rank_ic"].notna().sum() > 2:
        ic_sharpe_corr = corr_df[["rank_ic", "sharpe"]].corr().iloc[0, 1]
        ic_return_corr = corr_df[["rank_ic", "total_return"]].corr().iloc[0, 1]
        print(f"Rank IC ↔ Mean Sharpe:        r = {ic_sharpe_corr:.4f}")
        print(f"Rank IC ↔ Mean Total Return:  r = {ic_return_corr:.4f}")
        print("\nPer-model:")
        print(corr_df.round(5).to_string(index=False))
    corr_df.to_csv(OUTPUT_DIR / f"{run_name}_prediction_perf_correlation.csv", index=False)

    # Save
    result.aggregate_summary.reset_index().to_csv(
        OUTPUT_DIR / f"{run_name}_aggregate_summary.csv", index=False)
    result.fold_summary_long.to_csv(
        OUTPUT_DIR / f"{run_name}_fold_summary_long.csv", index=False)
    result.step_results.to_csv(
        OUTPUT_DIR / f"{run_name}_step_results.csv", index=False)

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
