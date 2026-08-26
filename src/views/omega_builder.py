from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd


def _annualize_std(
    std: pd.Series | float,
    horizon_days: int = 5,
    periods_per_year: int = 252,
):
    scale = np.sqrt(periods_per_year / horizon_days)
    return std * scale


def _annualize_var(
    var: pd.Series | float,
    horizon_days: int = 5,
    periods_per_year: int = 252,
):
    scale = periods_per_year / horizon_days
    return var * scale


def build_absolute_omega_from_uncertainty(
    meta_abs: pd.DataFrame,
    annualize: bool = False,
    horizon_days: int = 5,
    periods_per_year: int = 252,
    min_variance: float = 1e-8,
) -> pd.DataFrame:
    """
    absolute view의 uncertainty 컬럼으로 Omega 생성

    uncertainty는 std로 해석하고, diagonal variance로 변환
    """
    if "view_id" not in meta_abs.columns or "uncertainty" not in meta_abs.columns:
        raise ValueError("meta_abs에는 최소 view_id, uncertainty 컬럼이 있어야 합니다.")

    std = pd.to_numeric(meta_abs["uncertainty"], errors="coerce").fillna(np.nan)
    if annualize:
        std = _annualize_std(std, horizon_days=horizon_days, periods_per_year=periods_per_year)

    var = (std ** 2).fillna(np.nan)
    var = var.replace([np.inf, -np.inf], np.nan).fillna(min_variance)
    var = var.clip(lower=min_variance)

    Omega = pd.DataFrame(
        np.diag(var.values),
        index=meta_abs["view_id"].tolist(),
        columns=meta_abs["view_id"].tolist(),
    )
    return Omega


def build_relative_omega_from_uncertainty(
    meta_rel: pd.DataFrame,
    annualize: bool = False,
    horizon_days: int = 5,
    periods_per_year: int = 252,
    min_variance: float = 1e-8,
) -> pd.DataFrame:
    """
    relative view의 uncertainty로 Omega 생성

    보수적으로:
    Var(long-short) ≈ Var(long) + Var(short)
    """
    required = {"view_id", "long_uncertainty", "short_uncertainty"}
    missing = required - set(meta_rel.columns)
    if missing:
        raise ValueError(f"meta_rel에 필요한 컬럼이 없습니다: {missing}")

    long_std = pd.to_numeric(meta_rel["long_uncertainty"], errors="coerce")
    short_std = pd.to_numeric(meta_rel["short_uncertainty"], errors="coerce")

    if annualize:
        long_std = _annualize_std(long_std, horizon_days=horizon_days, periods_per_year=periods_per_year)
        short_std = _annualize_std(short_std, horizon_days=horizon_days, periods_per_year=periods_per_year)

    var = (long_std ** 2 + short_std ** 2).fillna(min_variance)
    var = var.replace([np.inf, -np.inf], np.nan).fillna(min_variance)
    var = var.clip(lower=min_variance)

    Omega = pd.DataFrame(
        np.diag(var.values),
        index=meta_rel["view_id"].tolist(),
        columns=meta_rel["view_id"].tolist(),
    )
    return Omega


def build_asset_error_statistics(
    per_prediction: pd.DataFrame,
    model_name: str,
    date: pd.Timestamp | str,
    fold_id: Optional[int] = None,
    evaluation_split: Optional[str] = None,
    window_days: int = 252,
    min_obs: int = 20,
) -> pd.DataFrame:
    """
    특정 시점 이전의 rolling error history로 asset별 오차 통계 생성

    Returns
    -------
    pd.DataFrame
        index=asset
        columns=[mae, rmse, bias, n_obs]
    """
    required = {"model", "date", "asset", "error"}
    missing = required - set(per_prediction.columns)
    if missing:
        raise ValueError(f"per_prediction에 필요한 컬럼이 없습니다: {missing}")

    df = per_prediction.copy()
    df["date"] = pd.to_datetime(df["date"])
    date = pd.Timestamp(date)

    df = df[df["model"] == model_name].copy()
    df = df[df["date"] < date].copy()

    if fold_id is not None and "fold_id" in df.columns:
        df = df[df["fold_id"] == fold_id].copy()

    if evaluation_split is not None and "evaluation_split" in df.columns:
        df = df[df["evaluation_split"] == evaluation_split].copy()

    df = df.sort_values("date")
    if window_days is not None and len(df) > 0:
        cutoff = date - pd.Timedelta(days=window_days * 2)  # 거래일 근사 여유
        df = df[df["date"] >= cutoff].copy()

    stats_rows = []
    for asset, g in df.groupby("asset"):
        e = pd.to_numeric(g["error"], errors="coerce").dropna()
        if len(e) < min_obs:
            continue

        stats_rows.append(
            {
                "asset": asset,
                "n_obs": len(e),
                "mae": float(e.abs().mean()),
                "rmse": float(np.sqrt((e ** 2).mean())),
                "bias": float(e.mean()),
            }
        )

    if len(stats_rows) == 0:
        return pd.DataFrame(columns=["n_obs", "mae", "rmse", "bias"]).set_index(pd.Index([], name="asset"))

    stats = pd.DataFrame(stats_rows).set_index("asset").sort_index()
    return stats


def build_absolute_omega_from_error_history(
    meta_abs: pd.DataFrame,
    asset_error_stats: pd.DataFrame,
    annualize: bool = False,
    horizon_days: int = 5,
    periods_per_year: int = 252,
    error_metric: str = "rmse",
    min_variance: float = 1e-8,
    fallback_variance: float = 1e-4,
) -> pd.DataFrame:
    """
    asset별 rolling error history로 absolute Omega 생성
    """
    required = {"view_id", "asset"}
    missing = required - set(meta_abs.columns)
    if missing:
        raise ValueError(f"meta_abs에 필요한 컬럼이 없습니다: {missing}")

    if error_metric not in {"rmse", "mae"}:
        raise ValueError("error_metric은 'rmse' 또는 'mae' 여야 합니다.")

    vals = []
    for _, row in meta_abs.iterrows():
        asset = row["asset"]
        if asset in asset_error_stats.index:
            std = float(asset_error_stats.loc[asset, error_metric])
            if annualize:
                std = float(_annualize_std(std, horizon_days=horizon_days, periods_per_year=periods_per_year))
            var = max(std ** 2, min_variance)
        else:
            var = fallback_variance
        vals.append(var)

    Omega = pd.DataFrame(
        np.diag(vals),
        index=meta_abs["view_id"].tolist(),
        columns=meta_abs["view_id"].tolist(),
    )
    return Omega


def build_relative_omega_from_error_history(
    meta_rel: pd.DataFrame,
    asset_error_stats: pd.DataFrame,
    annualize: bool = False,
    horizon_days: int = 5,
    periods_per_year: int = 252,
    error_metric: str = "rmse",
    min_variance: float = 1e-8,
    fallback_variance: float = 1e-4,
) -> pd.DataFrame:
    """
    asset별 rolling error history로 relative Omega 생성

    Var(long-short) ≈ Var(long) + Var(short)
    """
    required = {"view_id", "long_asset", "short_asset"}
    missing = required - set(meta_rel.columns)
    if missing:
        raise ValueError(f"meta_rel에 필요한 컬럼이 없습니다: {missing}")

    vals = []
    for _, row in meta_rel.iterrows():
        long_asset = row["long_asset"]
        short_asset = row["short_asset"]

        if long_asset in asset_error_stats.index:
            long_std = float(asset_error_stats.loc[long_asset, error_metric])
        else:
            long_std = np.sqrt(fallback_variance)

        if short_asset in asset_error_stats.index:
            short_std = float(asset_error_stats.loc[short_asset, error_metric])
        else:
            short_std = np.sqrt(fallback_variance)

        if annualize:
            long_std = float(_annualize_std(long_std, horizon_days=horizon_days, periods_per_year=periods_per_year))
            short_std = float(_annualize_std(short_std, horizon_days=horizon_days, periods_per_year=periods_per_year))

        var = max(long_std ** 2 + short_std ** 2, min_variance)
        vals.append(var)

    Omega = pd.DataFrame(
        np.diag(vals),
        index=meta_rel["view_id"].tolist(),
        columns=meta_rel["view_id"].tolist(),
    )
    return Omega


# =============================================================================
# Hybrid Omega: base variance × model quality × prediction strength
# =============================================================================
def _quality_multiplier(
    model_name: str,
    model_quality_scores: dict[str, float] | None,
    quality_floor: float = 0.005,
    quality_exponent: float = 0.5,
) -> float:
    """
    모델 품질(Rank IC 등)을 Omega 축소 factor로 변환.

    - model_quality_scores = {model_name: rank_ic} 형태 기대
    - rank_ic 높은 모델 → 작은 multiplier → 작은 Omega → 강한 신뢰
    - rank_ic 낮거나 음수 → 큰 multiplier → 큰 Omega → 불신

    음수 IC는 floor로 clip되어 "불신" 취급.
    """
    if not model_quality_scores or model_name not in model_quality_scores:
        return 1.0
    ic = float(model_quality_scores[model_name])
    # 음수 IC는 floor로 → 가장 불신하는 수준
    ic = max(ic, quality_floor)
    return float((1.0 / ic) ** quality_exponent)


def _strength_multiplier(
    q_values: np.ndarray,
    strength_floor: float = 1e-3,
    strength_exponent: float = 0.5,
) -> np.ndarray:
    """
    예측값 크기를 Omega 축소 factor로 변환.

    - |q_value| 큰 view → 작은 multiplier → 작은 Omega → 강한 신뢰
    - |q_value| 작은 view → 큰 multiplier → 큰 Omega → 약한 신뢰

    strength_exponent=0.5 이하로 설정하면 strength 영향을 제한적으로 반영.
    """
    abs_q = np.maximum(np.abs(q_values), strength_floor)
    return (1.0 / abs_q) ** strength_exponent


def build_absolute_omega_hybrid(
    meta_abs: pd.DataFrame,
    model_name: str,
    model_quality_scores: dict[str, float] | None = None,
    annualize: bool = False,
    horizon_days: int = 5,
    periods_per_year: int = 252,
    min_variance: float = 1e-8,
    max_variance: float = 1.0,
    quality_floor: float = 0.005,
    strength_floor: float = 1e-3,
    quality_exponent: float = 0.5,
    strength_exponent: float = 0.5,
    omega_scale: float = 1.0,
) -> pd.DataFrame:
    """
    Hybrid Omega = base_variance × quality_mult × strength_mult × omega_scale

    Components
    ----------
    - base_variance      : uncertainty² (모델 잔차 분산)
    - quality_mult       : 1 / rank_ic^q_exp   (모델 정확도 반영)
    - strength_mult      : 1 / |q_value|^s_exp (예측 확신도 반영)

    정확한 모델 + 강한 예측   → 작은 Omega → BL view 강 신뢰
    부정확한 모델 or 약한 예측 → 큰 Omega → BL market prior 유지
    """
    required = {"view_id", "uncertainty"}
    missing = required - set(meta_abs.columns)
    if missing:
        raise ValueError(f"meta_abs에 필요한 컬럼이 없습니다: {missing}")

    std = pd.to_numeric(meta_abs["uncertainty"], errors="coerce").fillna(np.nan)
    if annualize:
        std = _annualize_std(std, horizon_days=horizon_days, periods_per_year=periods_per_year)
    base_var = (std ** 2).fillna(min_variance)

    # model quality factor (scalar, same for all views from this model)
    q_mult = _quality_multiplier(
        model_name, model_quality_scores,
        quality_floor=quality_floor, quality_exponent=quality_exponent,
    )

    # prediction strength factor (per-view)
    if "q_value" in meta_abs.columns:
        q_vals = pd.to_numeric(meta_abs["q_value"], errors="coerce").fillna(0.0).values
        s_mult = _strength_multiplier(
            q_vals, strength_floor=strength_floor, strength_exponent=strength_exponent,
        )
    else:
        s_mult = np.ones(len(meta_abs))

    var = base_var.values * q_mult * s_mult * omega_scale
    var = np.where(np.isfinite(var), var, min_variance)
    var = np.clip(var, min_variance, max_variance)

    Omega = pd.DataFrame(
        np.diag(var),
        index=meta_abs["view_id"].tolist(),
        columns=meta_abs["view_id"].tolist(),
    )
    return Omega


def build_relative_omega_hybrid(
    meta_rel: pd.DataFrame,
    model_name: str,
    model_quality_scores: dict[str, float] | None = None,
    annualize: bool = False,
    horizon_days: int = 5,
    periods_per_year: int = 252,
    min_variance: float = 1e-8,
    max_variance: float = 1.0,
    quality_floor: float = 0.005,
    strength_floor: float = 1e-3,
    quality_exponent: float = 0.5,
    strength_exponent: float = 0.5,
    omega_scale: float = 1.0,
) -> pd.DataFrame:
    """
    Relative view용 Hybrid Omega.
    Var(long-short) ≈ Var(long) + Var(short) 을 base로 쓰고,
    동일한 quality/strength multiplier 적용.
    """
    required = {"view_id", "long_uncertainty", "short_uncertainty"}
    missing = required - set(meta_rel.columns)
    if missing:
        raise ValueError(f"meta_rel에 필요한 컬럼이 없습니다: {missing}")

    long_std = pd.to_numeric(meta_rel["long_uncertainty"], errors="coerce")
    short_std = pd.to_numeric(meta_rel["short_uncertainty"], errors="coerce")
    if annualize:
        long_std = _annualize_std(long_std, horizon_days=horizon_days, periods_per_year=periods_per_year)
        short_std = _annualize_std(short_std, horizon_days=horizon_days, periods_per_year=periods_per_year)
    base_var = (long_std ** 2 + short_std ** 2).fillna(min_variance)

    q_mult = _quality_multiplier(
        model_name, model_quality_scores,
        quality_floor=quality_floor, quality_exponent=quality_exponent,
    )

    # strength on the relative view's q_value (long - short prediction gap)
    if "q_value" in meta_rel.columns:
        q_vals = pd.to_numeric(meta_rel["q_value"], errors="coerce").fillna(0.0).values
        s_mult = _strength_multiplier(
            q_vals, strength_floor=strength_floor, strength_exponent=strength_exponent,
        )
    else:
        s_mult = np.ones(len(meta_rel))

    var = base_var.values * q_mult * s_mult * omega_scale
    var = np.where(np.isfinite(var), var, min_variance)
    var = np.clip(var, min_variance, max_variance)

    Omega = pd.DataFrame(
        np.diag(var),
        index=meta_rel["view_id"].tolist(),
        columns=meta_rel["view_id"].tolist(),
    )
    return Omega


def build_confidence_scaled_omega(
    P: pd.DataFrame,
    Sigma: pd.DataFrame,
    confidences: pd.Series | np.ndarray,
    tau: float = 0.05,
    min_variance: float = 1e-8,
) -> pd.DataFrame:
    """
    confidence score를 BL 스타일 Omega로 변환

    기본식
    ------
    base = diag(tau * P Σ P')
    confidence c in (0,1]
    Omega_i = base_i * (1-c)/c
    """
    if not P.index.equals(pd.Index(P.index)):
        P = P.copy()

    assets = list(P.columns)
    Sigma = Sigma.loc[assets, assets]

    base = tau * (P.values @ Sigma.values @ P.values.T)
    base_diag = np.diag(base)

    c = np.asarray(confidences, dtype=float).reshape(-1)
    if len(c) != len(P.index):
        raise ValueError("confidences 길이는 view 개수와 같아야 합니다.")

    c = np.clip(c, 1e-6, 1.0)
    omega_diag = base_diag * (1.0 - c) / c
    omega_diag = np.clip(omega_diag, min_variance, None)

    Omega = pd.DataFrame(
        np.diag(omega_diag),
        index=P.index.tolist(),
        columns=P.index.tolist(),
    )
    return Omega


def combine_omega_matrices(*omega_list: pd.DataFrame) -> pd.DataFrame:
    """
    여러 Omega를 block diagonal 형태로 결합
    """
    valid = [o for o in omega_list if o is not None and not o.empty]
    if len(valid) == 0:
        return pd.DataFrame()

    total_index = []
    for o in valid:
        total_index.extend(o.index.tolist())

    out = pd.DataFrame(0.0, index=total_index, columns=total_index)

    for o in valid:
        out.loc[o.index, o.columns] = o.values

    return out