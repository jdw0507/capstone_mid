from __future__ import annotations

"""
main_forecast.py
================
Improved forecasting pipeline — better features + tuned hyperparameters.

Changes vs default main_dqn_perfect.py forecasters:
  1. Derived features: log-return lags, MA ratios, volatility, cross-asset momentum
  2. Longer lookback for DL models (60→120)
  3. More training epochs for DL (10→30) with early stopping
  4. Larger hidden dims for DL models
  5. Better ML regularisation (tuned alpha, depth, estimators)
"""

from copy import deepcopy
from pathlib import Path
import time
import warnings

import numpy as np
import pandas as pd
import optuna
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
from scipy.stats import spearmanr

from src.data.csv_adapter import load_final_clean_result, build_asset_panel, make_model_matrices
from src.data.splitters import slice_by_split
from src.data.splitters import build_nested_walk_forward_splits, describe_split_coverage
from src.forecasting.targets import add_excess_target_to_panel
from src.forecasting.prediction_store import save_forecast_artifacts
from src.experiments.run_forecast_baselines import run_forecast_baselines_on_splits

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


# =============================================================================
# Config
# =============================================================================
DATASET_PATH     = "bl_v3_dataset.csv"
PERIODS_PER_YEAR = 252
HORIZON_DAYS     = 5
RISK_FREE_RATE   = 0.02

MODEL_TRAIN_DAYS = 1008
VIEW_BUILD_DAYS  = 252
TEST_DAYS        = 756
STEP_DAYS        = 63

OUTPUT_DIR       = Path("outputs/forecast_improved")
RUN_NAME         = "forecast_v2"

# Target: "target_5d" (raw) or "target_5d_excess" (rf-adjusted)
TARGET_COL       = "target_5d"


# =============================================================================
# Feature engineering — add derived features to panel
# =============================================================================
def add_derived_features(panel: pd.DataFrame) -> pd.DataFrame:
    """
    Add per-asset derived features to the long-format panel.
    All features are backward-looking (no look-ahead).
    All outputs are scale-normalised (ratios, changes, bounded indicators).
    """
    out = panel.sort_values(["asset", "date"]).copy()

    # ── 1. Convert macro levels → change rates (once per date) ───────
    #    Raw levels (VIX~10-80, MOVE~50-200, HY_OAS~300-2000) have
    #    incompatible scales. Replace with 1d pct_change + 5d pct_change.
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
        ma20   = series.rolling(20, min_periods=10).mean()
        ratio  = (series / ma20).rename(f"{col}_ma20ratio")
        derived = pd.concat([chg_1d, chg_5d, ratio], axis=1).reset_index()
        out = out.merge(derived, on="date", how="left")

    # Drop raw macro level columns (keep only change-rate versions)
    out = out.drop(columns=[c for c in macro_cols if c in out.columns], errors="ignore")

    # ── 2. Per-asset features ────────────────────────────────────────
    derived_parts = []
    for _, g in out.groupby("asset"):
        g = g.sort_values("date").copy()

        px = g["close"].astype(float)
        log_ret = np.log(px / px.shift(1))

        # Log-return lags (1d, 5d, 10d, 20d) — all ~N(0, 0.02)
        g["log_ret_1d"]  = log_ret
        g["log_ret_5d"]  = np.log(px / px.shift(5))
        g["log_ret_10d"] = np.log(px / px.shift(10))
        g["log_ret_20d"] = np.log(px / px.shift(20))

        # Realised volatility (5d, 20d) — ~0.005-0.05
        g["rvol_5d"]  = log_ret.rolling(5,  min_periods=3).std()
        g["rvol_20d"] = log_ret.rolling(20, min_periods=10).std()

        # Volatility ratio — ~0.5-2.0
        g["vol_ratio_5_20"] = g["rvol_5d"] / g["rvol_20d"].replace(0, np.nan)

        # MA ratios — ~0.9-1.1
        ma5  = px.rolling(5,  min_periods=3).mean()
        ma20 = px.rolling(20, min_periods=10).mean()
        ma60 = px.rolling(60, min_periods=30).mean()
        g["px_ma5_ratio"]   = px / ma5.replace(0, np.nan)
        g["px_ma20_ratio"]  = px / ma20.replace(0, np.nan)
        g["px_ma60_ratio"]  = px / ma60.replace(0, np.nan)
        g["ma5_ma20_ratio"] = ma5 / ma20.replace(0, np.nan)

        # Rolling skewness — ~-2 to 2
        g["ret_skew_20d"] = log_ret.rolling(20, min_periods=10).skew()

        # RSI change (momentum of momentum) — ~-30 to 30
        if "RSI_14" in g.columns:
            g["rsi_delta_5d"] = g["RSI_14"].astype(float).diff(5)

        # ATR regime (20d MA of ATR_14_pct) — ~0.5-5.0
        if "ATR_14_pct" in g.columns:
            g["atr_regime"] = g["ATR_14_pct"].astype(float).rolling(20, min_periods=10).mean()

        # DPO normalised by price — ~-0.05 to 0.05 (instead of raw price diff)
        if "DPO_20" in g.columns:
            g["DPO_20"] = g["DPO_20"].astype(float) / px.replace(0, np.nan)

        # MarketCap_Weight change — ~-0.01 to 0.01
        if "MarketCap_Weight" in g.columns:
            g["mcap_wt_chg5d"] = g["MarketCap_Weight"].astype(float).pct_change(5)

        derived_parts.append(g)

    out = pd.concat(derived_parts, ignore_index=True)

    # ── 3. Cross-sectional normalisation (per-date, across assets) ──
    #    For each date, compute:
    #      - rank (normalised 0..1): where this asset stands vs peers
    #      - zscore: standardised vs peers on that date
    #    Applied to features that are meaningful cross-sectionally.
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
        # Rank (0..1): higher rank = higher value vs peers on that date
        out[f"{col}_csrank"] = gb[col].transform(lambda s: s.rank(pct=True))
        # Cross-sectional zscore
        out[f"{col}_cszscore"] = gb[col].transform(
            lambda s: (s - s.mean()) / (s.std(ddof=0) + 1e-10)
        )

    # Replace inf/-inf
    out = out.replace([np.inf, -np.inf], np.nan)

    return out.sort_values(["asset", "date"]).reset_index(drop=True)


# =============================================================================
# Optuna hyperparameter tuning
# =============================================================================
def _eval_one_fold(forecaster, panel, split_row, target_col=TARGET_COL):
    """Train on model_train, evaluate RMSE on view_build for one fold."""
    parts = slice_by_split(panel, pd.Series(split_row._asdict()), date_col="date")
    train_df = parts["model_train"]
    eval_df = parts["view_build"]

    X_tr, y_tr, meta_tr = make_model_matrices(train_df, target_col=target_col, exclude_target_like=True)
    X_ev, y_ev, meta_ev = make_model_matrices(eval_df, target_col=target_col, exclude_target_like=True)

    model = deepcopy(forecaster)
    model.fit(X_tr, y_tr, meta_tr)
    pred = model.predict(X_ev, meta_ev)

    # Align indices — pred may have MultiIndex, y_ev may not (or vice versa)
    y_arr = y_ev.reset_index(drop=True).values.astype(float)
    p_arr = pred.reset_index(drop=True).values.astype(float)
    valid = np.isfinite(y_arr) & np.isfinite(p_arr)
    if valid.sum() < 10:
        return float("inf")
    return float(np.sqrt(np.mean((y_arr[valid] - p_arr[valid]) ** 2)))


def run_optuna_tuning(panel, splits, ml_trials=25, dl_trials=40, tune_folds=3,
                       only_models: set | None = None):
    """
    Tune each model type via Optuna on first `tune_folds` folds.
    DL models get more trials and wider search ranges.
    If `only_models` is provided, only those are tuned (others skipped).
    """
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    kw = dict(target_column=TARGET_COL, horizon_days=HORIZON_DAYS,
              periods_per_year=PERIODS_PER_YEAR)
    tune_splits = list(splits.head(tune_folds).itertuples(index=False))
    best_params = {}

    def _run(name, objective, n_trials):
        if only_models is not None and name not in only_models:
            return
        print(f"[Tune] {name} ({n_trials} trials)…")
        study = optuna.create_study(direction="minimize")
        study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
        best_params[name] = study.best_params
        print(f"  best RMSE={study.best_value:.6f} params={study.best_params}")

    # ── ML models ────────────────────────────────────────────────────
    def xgb_obj(trial):
        f = XGBoostForecaster(
            n_estimators=trial.suggest_int("n_estimators", 100, 1000, step=100),
            learning_rate=trial.suggest_float("lr", 0.003, 0.1, log=True),
            max_depth=trial.suggest_int("max_depth", 3, 10),
            **kw)
        return np.mean([_eval_one_fold(f, panel, s) for s in tune_splits])
    _run("XGBoost", xgb_obj, ml_trials)

    def lgb_obj(trial):
        f = LightGBMForecaster(
            n_estimators=trial.suggest_int("n_estimators", 100, 1000, step=100),
            learning_rate=trial.suggest_float("lr", 0.003, 0.1, log=True),
            num_leaves=trial.suggest_int("num_leaves", 7, 127),
            **kw)
        return np.mean([_eval_one_fold(f, panel, s) for s in tune_splits])
    _run("LightGBM", lgb_obj, ml_trials)

    def cat_obj(trial):
        f = CatBoostForecaster(
            iterations=trial.suggest_int("iterations", 100, 1000, step=100),
            learning_rate=trial.suggest_float("lr", 0.003, 0.1, log=True),
            depth=trial.suggest_int("depth", 3, 10),
            **kw)
        return np.mean([_eval_one_fold(f, panel, s) for s in tune_splits])
    _run("CatBoost", cat_obj, ml_trials)

    def rf_obj(trial):
        f = RandomForestForecaster(
            n_estimators=trial.suggest_int("n_estimators", 100, 1000, step=100),
            max_depth=trial.suggest_int("max_depth", 4, 16),
            min_samples_leaf=trial.suggest_int("min_samples_leaf", 3, 60),
            **kw)
        return np.mean([_eval_one_fold(f, panel, s) for s in tune_splits])
    _run("RandomForest", rf_obj, ml_trials)

    # ── DL models — wider ranges, more trials ────────────────────────
    def lstm_obj(trial):
        f = LSTMForecaster(
            lookback=trial.suggest_categorical("lookback", [30, 60, 90, 120, 180]),
            hidden_dim=trial.suggest_categorical("hidden_dim", [32, 64, 128, 256]),
            num_layers=trial.suggest_int("num_layers", 1, 4),
            lr=trial.suggest_float("lr", 5e-5, 3e-3, log=True),
            dropout=trial.suggest_float("dropout", 0.0, 0.4),
            epochs=trial.suggest_categorical("epochs", [15, 25, 40]),
            batch_size=trial.suggest_categorical("batch_size", [128, 256, 512]),
            verbose=False, **kw)
        return np.mean([_eval_one_fold(f, panel, s) for s in tune_splits])
    _run("LSTM", lstm_obj, dl_trials)

    def tf_obj(trial):
        d_model = trial.suggest_categorical("d_model", [64, 128, 192, 256])
        n_heads = trial.suggest_categorical("n_heads", [2, 4, 8, 16])
        if d_model % n_heads != 0:
            raise optuna.TrialPruned()
        f = TransformerForecaster(
            lookback=trial.suggest_categorical("lookback", [30, 60, 90, 120, 180, 240]),
            d_model=d_model, n_heads=n_heads,
            num_layers=trial.suggest_int("num_layers", 1, 5),
            lr=trial.suggest_float("lr", 3e-5, 5e-3, log=True),
            dropout=trial.suggest_float("dropout", 0.0, 0.5),
            epochs=trial.suggest_categorical("epochs", [15, 25, 40, 60]),
            batch_size=trial.suggest_categorical("batch_size", [128, 256, 512]),
            verbose=False, **kw)
        return np.mean([_eval_one_fold(f, panel, s) for s in tune_splits])
    _run("Transformer", tf_obj, dl_trials)

    def ptst_obj(trial):
        lookback = trial.suggest_categorical("lookback", [30, 60, 90, 120, 180])
        patch_len = trial.suggest_categorical("patch_len", [5, 10, 15, 20, 30])
        stride = trial.suggest_categorical("stride", [3, 5, 10, 15])
        if patch_len > lookback or stride > patch_len:
            raise optuna.TrialPruned()
        f = PatchTSTForecaster(
            lookback=lookback, patch_len=patch_len, stride=stride,
            lr=trial.suggest_float("lr", 5e-5, 3e-3, log=True),
            dropout=trial.suggest_float("dropout", 0.0, 0.4),
            epochs=trial.suggest_categorical("epochs", [15, 25, 40]),
            batch_size=trial.suggest_categorical("batch_size", [128, 256, 512]),
            verbose=False, **kw)
        return np.mean([_eval_one_fold(f, panel, s) for s in tune_splits])
    _run("PatchTST", ptst_obj, dl_trials)

    def cnn_obj(trial):
        # Expanded CNN: deeper + wider channels
        n_ch = trial.suggest_int("n_channels", 2, 5)
        dims = []
        for i in range(n_ch):
            dims.append(trial.suggest_categorical(f"ch_{i}", [64, 128, 256, 384, 512]))
        f = CNN1DForecaster(
            lookback=trial.suggest_categorical("lookback", [30, 60, 90, 120, 180]),
            channels=tuple(dims),
            kernel_size=trial.suggest_categorical("kernel_size", [3, 5, 7, 9, 11]),
            lr=trial.suggest_float("lr", 5e-5, 5e-3, log=True),
            dropout=trial.suggest_float("dropout", 0.0, 0.5),
            epochs=trial.suggest_categorical("epochs", [15, 25, 40, 60]),
            batch_size=trial.suggest_categorical("batch_size", [128, 256, 512]),
            verbose=False, **kw)
        return np.mean([_eval_one_fold(f, panel, s) for s in tune_splits])
    _run("CNN1D", cnn_obj, dl_trials)

    def mlp_obj(trial):
        n_layers = trial.suggest_int("n_layers", 2, 5)
        dims = []
        for i in range(n_layers):
            dims.append(trial.suggest_categorical(f"dim_{i}", [64, 128, 256, 512, 1024]))
        f = MLPForecaster(
            lookback=trial.suggest_categorical("lookback", [30, 60, 90, 120]),
            hidden_dims=tuple(dims),
            lr=trial.suggest_float("lr", 5e-5, 3e-3, log=True),
            dropout=trial.suggest_float("dropout", 0.0, 0.5),
            epochs=trial.suggest_categorical("epochs", [15, 25, 40, 60]),
            batch_size=trial.suggest_categorical("batch_size", [128, 256, 512]),
            verbose=False, **kw)
        return np.mean([_eval_one_fold(f, panel, s) for s in tune_splits])
    _run("MLP", mlp_obj, dl_trials)

    def hybrid_obj(trial):
        d_model = trial.suggest_categorical("d_model", [32, 64, 128])
        n_heads = trial.suggest_categorical("n_heads", [2, 4, 8])
        if d_model % n_heads != 0:
            raise optuna.TrialPruned()
        f = HybridLSTMTransformerForecaster(
            lookback=trial.suggest_categorical("lookback", [30, 60, 90, 120, 180]),
            hidden_dim=trial.suggest_categorical("hidden_dim", [32, 64, 128, 256]),
            lstm_layers=trial.suggest_int("lstm_layers", 1, 3),
            d_model=d_model, n_heads=n_heads,
            transformer_layers=trial.suggest_int("tf_layers", 1, 3),
            lr=trial.suggest_float("lr", 5e-5, 3e-3, log=True),
            dropout=trial.suggest_float("dropout", 0.0, 0.4),
            epochs=trial.suggest_categorical("epochs", [15, 25, 40]),
            batch_size=trial.suggest_categorical("batch_size", [128, 256, 512]),
            verbose=False, **kw)
        return np.mean([_eval_one_fold(f, panel, s) for s in tune_splits])
    _run("HybridLSTMTF", hybrid_obj, dl_trials)

    return best_params


# =============================================================================
# Build forecasters from tuned params
# =============================================================================
def build_tuned_forecasters(best_params: dict) -> dict[str, object]:
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

    def _dl_kw(params):
        """Extract common DL kwargs from tuned params with defaults."""
        return dict(
            epochs=params.get("epochs", 30),
            batch_size=params.get("batch_size", 512),
            verbose=False, **kw,
        )

    # CNN1D channels reconstruction (handles both old ch1/ch2 format and new ch_i format)
    cnn_n = cnn.get("n_channels", 3)
    if "ch_0" in cnn:  # new format
        cnn_channels = tuple(cnn.get(f"ch_{i}", 64) for i in range(cnn_n))
    elif "ch1" in cnn:  # old format (n_channels = 2 or 3)
        ch1, ch2 = cnn.get("ch1", 64), cnn.get("ch2", 128)
        cnn_channels = (ch1, ch2) if cnn_n == 2 else (ch1, ch2, ch1)
    else:
        cnn_channels = (64, 128, 64)

    # MLP hidden_dims reconstruction
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
        "HybridLSTMTF":   HybridLSTMTransformerForecaster(
                              lookback=hyb.get("lookback", lstm.get("lookback", 60)),
                              hidden_dim=hyb.get("hidden_dim", lstm.get("hidden_dim", 128)),
                              lstm_layers=hyb.get("lstm_layers", 2),
                              d_model=hyb.get("d_model", tf.get("d_model", 128)),
                              n_heads=hyb.get("n_heads", tf.get("n_heads", 8)),
                              transformer_layers=hyb.get("tf_layers", 2),
                              lr=hyb.get("lr", tf.get("lr", 3e-4)),
                              dropout=hyb.get("dropout", tf.get("dropout", 0.15)),
                              **_dl_kw(hyb)),
    }


# =============================================================================
# Metrics (sklearn / scipy)
# =============================================================================
def calc_wmape(y_true, y_pred):
    """Weighted MAPE: sum(|err|) / sum(|y_true|) — robust to 0 targets."""
    yt, yp = np.asarray(y_true, float), np.asarray(y_pred, float)
    denom = np.sum(np.abs(yt))
    if denom < 1e-12:
        return float("nan")
    return float(np.sum(np.abs(yt - yp)) / denom * 100)

def calc_mae(y_true, y_pred):
    return float(mean_absolute_error(y_true, y_pred))

def calc_rmse(y_true, y_pred):
    return float(root_mean_squared_error(y_true, y_pred))

def calc_direction_accuracy(y_true, y_pred):
    yt, yp = np.asarray(y_true, float), np.asarray(y_pred, float)
    return float(np.mean(np.sign(yt) == np.sign(yp)))

def calc_rank_ic(y_true, y_pred):
    corr, _ = spearmanr(y_true, y_pred, nan_policy="omit")
    return float(corr) if np.isfinite(corr) else np.nan


# =============================================================================
# Main
# =============================================================================
def run_experiment(csv_path: str) -> None:
    dataset_tag = Path(csv_path).stem.lower()

    print("\n" + "=" * 90)
    print(f"=== IMPROVED FORECASTING: {csv_path} ===")
    print("=" * 90)

    # ── 1. Load data ─────────────────────────────────────────────────
    raw_df = load_final_clean_result(csv_path, date_col="Date")
    panel = build_asset_panel(
        raw_df, target_prefix="Return_5d_", target_name="target_5d",
        include_shared=True, include_close=True, dropna_target=True)

    print(f"[INFO] Raw panel: {len(panel)} rows, {panel['asset'].nunique()} assets")
    print(f"[INFO] Base features: {[c for c in panel.columns if c not in ['date','asset']]}")

    # ── 2. Feature engineering ───────────────────────────────────────
    print("\n[INFO] Adding derived features…")
    panel = add_derived_features(panel)

    # Add excess return column only if we want to use it as target
    if TARGET_COL == "target_5d_excess":
        panel = add_excess_target_to_panel(
            panel, target_col="target_5d", output_col="target_5d_excess",
            risk_free_rate=RISK_FREE_RATE, horizon_days=HORIZON_DAYS,
            periods_per_year=PERIODS_PER_YEAR, rf_is_annualized=True)
        print(f"[INFO] Target: {TARGET_COL} (raw 5d return − rf×5/252)")
    else:
        print(f"[INFO] Target: {TARGET_COL} (raw 5d return, no rf adjustment)")

    # Drop raw-scale columns that cause ill-conditioning in linear models
    # MarketCap_USD (billions), close (raw price) — use ratios/weights instead
    drop_cols = {"date", "asset", "target_5d", "target_5d_excess",
                 "close", "MarketCap_USD"}
    feature_cols = [c for c in panel.columns
                    if c not in drop_cols
                    and "target" not in c.lower() and "return" not in c.lower()
                    and "future" not in c.lower()]
    # Drop close and MarketCap_USD from the panel so they don't leak into model matrices
    panel = panel.drop(columns=[c for c in ["close", "MarketCap_USD"] if c in panel.columns], errors="ignore")
    print(f"[INFO] Total features: {len(feature_cols)}")
    print(f"[INFO] New features: {[c for c in feature_cols if c not in ['close','RSI_14','StochRSI_14','ROC_10','TSI_25_13','DPO_20','ATR_14_pct','MarketCap_USD','MarketCap_Weight','VIX','MOVE','HY_OAS','Spread_10Y2Y','Dollar_Index']]}")

    # ── 3. Splits ────────────────────────────────────────────────────
    splits = build_nested_walk_forward_splits(
        panel, model_train_days=MODEL_TRAIN_DAYS, view_build_days=VIEW_BUILD_DAYS,
        test_days=TEST_DAYS, step_days=STEP_DAYS, date_col="date")
    print(f"\n{len(splits)} folds")
    print(describe_split_coverage(splits).to_string(index=False))

    # ── 4. Optuna tuning → build forecasters ────────────────────────
    params_path = OUTPUT_DIR / f"{RUN_NAME}_best_params.csv"
    RETUNE_MODELS = {"CNN1D", "Transformer", "MLP"}  # selective retune
    RETUNE_TRIALS = 60

    def _load_params(path):
        out = {}
        if not path.exists():
            return out
        df = pd.read_csv(path)
        for model, g in df.groupby("model"):
            out[model] = {}
            for _, row in g.iterrows():
                v = row["value"]
                if isinstance(v, float) and v.is_integer():
                    v = int(v)
                out[model][row["param"]] = v
        return out

    def _save_params(params, path):
        df = pd.DataFrame([
            {"model": k, "param": pk, "value": pv}
            for k, ps in params.items() for pk, pv in ps.items()
        ])
        df.to_csv(path, index=False)

    best_params = _load_params(params_path)
    missing = [m for m in ["XGBoost", "LightGBM", "CatBoost", "RandomForest",
                            "LSTM", "Transformer", "PatchTST", "CNN1D", "MLP", "HybridLSTMTF"]
               if m not in best_params]

    if missing:
        print(f"\n=== Full tuning (missing: {missing}) ===")
        new_params = run_optuna_tuning(panel, splits, ml_trials=25, dl_trials=40, tune_folds=3)
        best_params.update(new_params)
        _save_params(best_params, params_path)
    else:
        print(f"\n[INFO] Loaded cached params for {len(best_params)} models")

    # Selective retune of underperforming DL models (wider search, more trials)
    print(f"\n=== Selective retune ({RETUNE_TRIALS} trials): {RETUNE_MODELS} ===")
    retuned = run_optuna_tuning(
        panel, splits, ml_trials=RETUNE_TRIALS, dl_trials=RETUNE_TRIALS,
        tune_folds=3, only_models=RETUNE_MODELS,
    )
    best_params.update(retuned)
    _save_params(best_params, params_path)
    print(f"[INFO] Updated params → {params_path}")

    forecasters = build_tuned_forecasters(best_params)
    model_names = list(forecasters.keys())
    print(f"\nTuned models ({len(model_names)}): {model_names}")

    # ── 5. Run forecasts ─────────────────────────────────────────────
    print("\n=== Running view_build forecasts ===")
    r_view = run_forecast_baselines_on_splits(
        panel_df=panel, splits=splits, forecasters=forecasters,
        target_col=TARGET_COL, evaluation_split="view_build",
        exclude_target_like=True, date_col="date", verbose_progress=True)

    print("\n=== Running test forecasts ===")
    r_test = run_forecast_baselines_on_splits(
        panel_df=panel, splits=splits, forecasters=forecasters,
        target_col=TARGET_COL, evaluation_split="test",
        exclude_target_like=True, date_col="date", verbose_progress=True)

    per_pred = pd.concat([r_view.per_prediction, r_test.per_prediction], ignore_index=True)

    # ── 6. Save ──────────────────────────────────────────────────────
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
        output_dir=OUTPUT_DIR, run_name=RUN_NAME, save_csv=True, save_parquet=True)

    # ── 7. Evaluate ──────────────────────────────────────────────────
    print("\n" + "=" * 90)
    print("=== RESULTS (test set) ===")
    print("=" * 90)

    test_pred = per_pred[per_pred["evaluation_split"] == "test"].copy()
    metrics = []
    for model, g in test_pred.groupby("model"):
        yt, yp = g["y_true"].astype(float), g["y_pred"].astype(float)
        valid = yt.notna() & yp.notna()
        yt, yp = yt[valid], yp[valid]
        metrics.append({
            "model": model,
            "wmape_pct": calc_wmape(yt, yp),
            "mae": calc_mae(yt, yp),
            "rmse": calc_rmse(yt, yp),
            "dir_acc": calc_direction_accuracy(yt, yp),
            "rank_ic": calc_rank_ic(yt, yp),
            "n": len(yt),
        })

    metrics_df = pd.DataFrame(metrics).sort_values("wmape_pct")
    print(metrics_df.round(4).to_string(index=False))
    metrics_df.to_csv(OUTPUT_DIR / f"{RUN_NAME}_test_metrics.csv", index=False)

    # ── Ensemble: RMSE-inverse weighted average of predictions ─────────
    print("\n=== Ensembles ===")
    view_pred = per_pred[per_pred["evaluation_split"] == "view_build"].copy()

    # Weights computed on view_build (no test leakage)
    view_rmse = {}
    for model, g in view_pred.groupby("model"):
        yt, yp = g["y_true"].astype(float), g["y_pred"].astype(float)
        v = yt.notna() & yp.notna()
        if v.sum() > 10:
            view_rmse[model] = calc_rmse(yt[v], yp[v])

    def _ensemble_preds(model_list, weights_dict, split_df, name):
        """Build weighted-average predictions for given models."""
        sub = split_df[split_df["model"].isin(model_list)].copy()
        if sub.empty:
            return None
        # Pivot: index=(date,asset,fold_id), columns=model, values=y_pred
        piv = sub.pivot_table(
            index=["fold_id", "date", "asset"], columns="model", values="y_pred",
            aggfunc="mean",
        )
        truth = sub.groupby(["fold_id", "date", "asset"])["y_true"].first()
        # Keep only requested models present in pivot
        cols = [m for m in model_list if m in piv.columns]
        w = np.array([1.0 / max(weights_dict.get(m, 1e-6), 1e-6) for m in cols])
        w = w / w.sum()
        # Fill NaNs per-row with the row mean (numpy-based for safety)
        arr = piv[cols].to_numpy(dtype=float)
        row_means = np.nanmean(arr, axis=1, keepdims=True)
        # Fallback: if a row is entirely NaN, row_mean is NaN → set to 0
        row_means = np.where(np.isnan(row_means), 0.0, row_means)
        nan_mask = np.isnan(arr)
        arr = np.where(nan_mask, np.broadcast_to(row_means, arr.shape), arr)
        ens_pred = pd.Series((arr * w).sum(axis=1), index=piv.index)
        y_true = truth.reindex(ens_pred.index)
        valid = y_true.notna() & ens_pred.notna()
        if valid.sum() < 10:
            return None
        return {
            "model": name,
            "wmape_pct": calc_wmape(y_true[valid], ens_pred[valid]),
            "mae": calc_mae(y_true[valid], ens_pred[valid]),
            "rmse": calc_rmse(y_true[valid], ens_pred[valid]),
            "dir_acc": calc_direction_accuracy(y_true[valid], ens_pred[valid]),
            "rank_ic": calc_rank_ic(y_true[valid], ens_pred[valid]),
            "n": int(valid.sum()),
        }

    ensembles = []
    top5_models = metrics_df.head(5)["model"].tolist()
    top3_models = metrics_df.head(3)["model"].tolist()
    dl_models = [m for m in ["LSTM", "Transformer", "PatchTST", "CNN1D", "MLP", "HybridLSTMTF"]
                 if m in view_rmse]
    ml_models = [m for m in ["XGBoost", "CatBoost", "LightGBM", "RandomForest", "Lasso"]
                 if m in view_rmse]
    all_models = list(view_rmse.keys())

    for name, models in [
        ("ENS_top3", top3_models),
        ("ENS_top5", top5_models),
        ("ENS_ML", ml_models),
        ("ENS_DL", dl_models),
        ("ENS_All", all_models),
    ]:
        r = _ensemble_preds(models, view_rmse, test_pred, name)
        if r is not None:
            ensembles.append(r)
            print(f"  {name}: WMAPE={r['wmape_pct']:.2f}% MAE={r['mae']:.5f} "
                  f"RMSE={r['rmse']:.4f} DirAcc={r['dir_acc']:.4f} "
                  f"RankIC={r['rank_ic']:.4f} ({len(models)} models)")

    if ensembles:
        ens_df = pd.DataFrame(ensembles)
        combined_df = pd.concat([metrics_df, ens_df], ignore_index=True).sort_values("wmape_pct")
        print("\n=== Combined (models + ensembles) ===")
        print(combined_df.round(4).to_string(index=False))
        combined_df.to_csv(OUTPUT_DIR / f"{RUN_NAME}_metrics_with_ensembles.csv", index=False)

    # Per-fold breakdown for best 3 models
    print("\n=== Per-fold MAPE for top-3 models ===")
    top3 = metrics_df.head(3)["model"].tolist()
    fold_metrics = []
    for (model, fold_id), g in test_pred[test_pred["model"].isin(top3)].groupby(["model", "fold_id"]):
        yt, yp = g["y_true"].astype(float), g["y_pred"].astype(float)
        valid = yt.notna() & yp.notna()
        fold_metrics.append({
            "model": model, "fold_id": fold_id,
            "wmape_pct": calc_wmape(yt[valid], yp[valid]),
            "dir_acc": calc_direction_accuracy(yt[valid], yp[valid]),
        })
    fold_df = pd.DataFrame(fold_metrics)
    pivot = fold_df.pivot_table(index="fold_id", columns="model", values="wmape_pct")
    print(pivot.round(2).to_string())
    fold_df.to_csv(OUTPUT_DIR / f"{RUN_NAME}_fold_metrics_top3.csv", index=False)

    print(f"\n=== Output files ===")
    for f in sorted(OUTPUT_DIR.glob(f"{RUN_NAME}_*")):
        print(f"  {f}")


def main():
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 320)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    run_experiment(DATASET_PATH)
    print("\n=== Done ===")


if __name__ == "__main__":
    main()
