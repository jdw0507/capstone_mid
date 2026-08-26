from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd


@dataclass
class ForecastMetricResult:
    per_row: pd.DataFrame
    summary_by_model: pd.DataFrame
    summary_by_model_asset: pd.DataFrame
    rank_ic_by_date: pd.DataFrame


def _to_series(x, name: str) -> pd.Series:
    if isinstance(x, pd.Series):
        s = x.copy()
        s.name = name
        return s

    arr = np.asarray(x).reshape(-1)
    return pd.Series(arr, name=name)


def mean_absolute_error(y_true: pd.Series, y_pred: pd.Series) -> float:
    df = pd.concat([y_true.rename("y_true"), y_pred.rename("y_pred")], axis=1).dropna()
    if df.empty:
        return np.nan
    return float((df["y_true"] - df["y_pred"]).abs().mean())


def root_mean_squared_error(y_true: pd.Series, y_pred: pd.Series) -> float:
    df = pd.concat([y_true.rename("y_true"), y_pred.rename("y_pred")], axis=1).dropna()
    if df.empty:
        return np.nan
    return float(np.sqrt(((df["y_true"] - df["y_pred"]) ** 2).mean()))


def bias(y_true: pd.Series, y_pred: pd.Series) -> float:
    df = pd.concat([y_true.rename("y_true"), y_pred.rename("y_pred")], axis=1).dropna()
    if df.empty:
        return np.nan
    return float((df["y_pred"] - df["y_true"]).mean())


def sign_accuracy(y_true: pd.Series, y_pred: pd.Series) -> float:
    df = pd.concat([y_true.rename("y_true"), y_pred.rename("y_pred")], axis=1).dropna()
    if df.empty:
        return np.nan

    true_sign = np.sign(df["y_true"])
    pred_sign = np.sign(df["y_pred"])
    return float((true_sign == pred_sign).mean())


def spearman_rank_ic(y_true: pd.Series, y_pred: pd.Series) -> float:
    df = pd.concat([y_true.rename("y_true"), y_pred.rename("y_pred")], axis=1).dropna()
    if len(df) < 3:
        return np.nan

    true_rank = df["y_true"].rank(method="average")
    pred_rank = df["y_pred"].rank(method="average")

    if true_rank.nunique() < 2 or pred_rank.nunique() < 2:
        return np.nan

    return float(true_rank.corr(pred_rank, method="pearson"))


def pearson_ic(y_true: pd.Series, y_pred: pd.Series) -> float:
    df = pd.concat([y_true.rename("y_true"), y_pred.rename("y_pred")], axis=1).dropna()
    if len(df) < 3:
        return np.nan

    if df["y_true"].nunique() < 2 or df["y_pred"].nunique() < 2:
        return np.nan

    return float(df["y_true"].corr(df["y_pred"], method="pearson"))


def build_forecast_evaluation_frame(
    y_true: pd.Series,
    y_pred: pd.Series,
    model_name: str,
    meta: Optional[pd.DataFrame] = None,
    uncertainty: Optional[pd.Series] = None,
) -> pd.DataFrame:
    y_true = _to_series(y_true, "y_true").reset_index(drop=True)
    y_pred = _to_series(y_pred, "y_pred").reset_index(drop=True)

    if len(y_true) != len(y_pred):
        raise ValueError(f"y_true와 y_pred 길이가 다릅니다. {len(y_true)} vs {len(y_pred)}")

    out = pd.DataFrame({
        "y_true": y_true,
        "y_pred": y_pred,
    })

    if meta is not None:
        if len(meta) != len(out):
            raise ValueError(f"meta와 예측 길이가 다릅니다. {len(meta)} vs {len(out)}")
        meta = meta.reset_index(drop=True).copy()
        out = pd.concat([meta, out], axis=1)

    if uncertainty is not None:
        uncertainty = _to_series(uncertainty, "uncertainty").reset_index(drop=True)
        if len(uncertainty) != len(out):
            raise ValueError(f"uncertainty 길이가 다릅니다. {len(uncertainty)} vs {len(out)}")
        out["uncertainty"] = uncertainty

    out["model"] = model_name
    out["valid_pair"] = out["y_true"].notna() & out["y_pred"].notna()

    out["error"] = np.where(out["valid_pair"], out["y_pred"] - out["y_true"], np.nan)
    out["abs_error"] = np.where(out["valid_pair"], np.abs(out["error"]), np.nan)
    out["sq_error"] = np.where(out["valid_pair"], out["error"] ** 2, np.nan)

    out["true_sign"] = np.where(out["valid_pair"], np.sign(out["y_true"]), np.nan)
    out["pred_sign"] = np.where(out["valid_pair"], np.sign(out["y_pred"]), np.nan)
    out["correct_sign"] = np.where(out["valid_pair"], out["true_sign"] == out["pred_sign"], np.nan)

    return out


def summarize_forecast_metrics(
    eval_df: pd.DataFrame,
) -> ForecastMetricResult:
    required = {"model", "y_true", "y_pred", "valid_pair"}
    missing = required - set(eval_df.columns)
    if missing:
        raise ValueError(f"필수 컬럼이 없습니다: {missing}")

    df = eval_df.copy()

    summary_by_model = (
        df.groupby("model")
        .agg(
            n_obs=("valid_pair", "sum"),
            mae=("abs_error", "mean"),
            rmse=("sq_error", lambda x: float(np.sqrt(np.nanmean(x))) if np.isfinite(np.nanmean(x)) else np.nan),
            bias=("error", "mean"),
            sign_accuracy=("correct_sign", "mean"),
            pred_mean=("y_pred", "mean"),
            true_mean=("y_true", "mean"),
        )
        .sort_values("mae", ascending=True)
    )

    if "asset" in df.columns:
        summary_by_model_asset = (
            df.groupby(["model", "asset"])
            .agg(
                n_obs=("valid_pair", "sum"),
                mae=("abs_error", "mean"),
                rmse=("sq_error", lambda x: float(np.sqrt(np.nanmean(x))) if np.isfinite(np.nanmean(x)) else np.nan),
                bias=("error", "mean"),
                sign_accuracy=("correct_sign", "mean"),
            )
            .sort_index()
        )
    else:
        summary_by_model_asset = pd.DataFrame()

    if {"date", "asset"}.issubset(df.columns):
        rank_ic_rows = []
        valid_df = df[df["valid_pair"]].copy()

        for (model, date), g in valid_df.groupby(["model", "date"]):
            rank_ic_rows.append(
                {
                    "model": model,
                    "date": date,
                    "rank_ic": spearman_rank_ic(g["y_true"], g["y_pred"]),
                    "pearson_ic": pearson_ic(g["y_true"], g["y_pred"]),
                    "n_assets": len(g),
                }
            )

        rank_ic_by_date = pd.DataFrame(rank_ic_rows).sort_values(["model", "date"])

        if not rank_ic_by_date.empty:
            rank_ic_summary = (
                rank_ic_by_date.groupby("model")
                .agg(
                    mean_rank_ic=("rank_ic", "mean"),
                    median_rank_ic=("rank_ic", "median"),
                    std_rank_ic=("rank_ic", "std"),
                    mean_pearson_ic=("pearson_ic", "mean"),
                )
            )
            summary_by_model = summary_by_model.join(rank_ic_summary, how="left")
    else:
        rank_ic_by_date = pd.DataFrame()

    return ForecastMetricResult(
        per_row=df,
        summary_by_model=summary_by_model,
        summary_by_model_asset=summary_by_model_asset,
        rank_ic_by_date=rank_ic_by_date,
    )