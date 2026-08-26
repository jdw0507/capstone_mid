from __future__ import annotations

from typing import Iterable, Optional, Sequence

import numpy as np
import pandas as pd


def annual_to_period_return(
    annual_rate: float | pd.Series,
    horizon_days: int,
    periods_per_year: int = 252,
) -> float | pd.Series:
    """
    연율 수익률/금리를 horizon 기간 수익률로 변환

    예:
    - annual_rate = 0.02
    - horizon_days = 5
    - periods_per_year = 252

    => 약 5거래일 기준 무위험수익률
    """
    return (1.0 + annual_rate) ** (horizon_days / periods_per_year) - 1.0


def period_to_annual_return(
    period_return: float | pd.Series,
    horizon_days: int,
    periods_per_year: int = 252,
) -> float | pd.Series:
    """
    horizon 기간 수익률을 연율 수익률로 변환
    """
    return (1.0 + period_return) ** (periods_per_year / horizon_days) - 1.0


def make_forward_simple_return_from_prices(
    prices: pd.Series | pd.DataFrame,
    horizon_days: int = 5,
    annualize: bool = False,
    periods_per_year: int = 252,
) -> pd.Series | pd.DataFrame:
    """
    가격 데이터로부터 미래 단순수익률 생성

    r_{t -> t+h} = P_{t+h} / P_t - 1

    Parameters
    ----------
    prices : pd.Series or pd.DataFrame
        가격 시계열
    horizon_days : int
        미래 예측 horizon
    annualize : bool
        True면 결과를 연율화
    periods_per_year : int
        일별이면 252, 월별이면 12
    """
    out = prices.shift(-horizon_days) / prices - 1.0

    if annualize:
        out = period_to_annual_return(out, horizon_days=horizon_days, periods_per_year=periods_per_year)

    return out


def make_forward_log_return_from_prices(
    prices: pd.Series | pd.DataFrame,
    horizon_days: int = 5,
    annualize: bool = False,
    periods_per_year: int = 252,
) -> pd.Series | pd.DataFrame:
    """
    가격 데이터로부터 미래 로그수익률 생성
    """
    out = np.log(prices.shift(-horizon_days) / prices)

    if annualize:
        out = out * (periods_per_year / horizon_days)

    return out


def add_forward_targets_to_wide_df(
    df: pd.DataFrame,
    assets: Sequence[str],
    close_prefix: str = "Close_",
    horizons: Iterable[int] = (5, 20, 60),
    periods_per_year: int = 252,
    annualize: bool = False,
) -> pd.DataFrame:
    """
    wide DataFrame에 자산별 forward return target 추가

    생성 컬럼 예:
    - Return_5d_AAPL
    - Return_20d_AAPL
    - Return_60d_AAPL
    """
    out = df.copy()

    for asset in assets:
        close_col = f"{close_prefix}{asset}"
        if close_col not in out.columns:
            continue

        px = pd.to_numeric(out[close_col], errors="coerce")

        for h in horizons:
            target_col = f"Return_{h}d_{asset}"
            out[target_col] = make_forward_simple_return_from_prices(
                px,
                horizon_days=h,
                annualize=annualize,
                periods_per_year=periods_per_year,
            )

    return out


def make_excess_return_target_from_series(
    target_return: pd.Series,
    risk_free_rate: float | pd.Series | None = None,
    horizon_days: int = 5,
    periods_per_year: int = 252,
    rf_is_annualized: bool = True,
) -> pd.Series:
    """
    단일 타깃 시계열을 excess return으로 변환

    Parameters
    ----------
    target_return : pd.Series
        미래 수익률 타깃
    risk_free_rate : float or pd.Series or None
        무위험수익률
        - float이면 상수
        - Series이면 시계열
    horizon_days : int
        타깃 horizon
    periods_per_year : int
        연율화 기준
    rf_is_annualized : bool
        risk_free_rate가 연율 기준인지 여부
    """
    y = pd.to_numeric(target_return, errors="coerce").copy()

    if risk_free_rate is None:
        return y

    if isinstance(risk_free_rate, pd.Series):
        rf = pd.to_numeric(risk_free_rate, errors="coerce").copy()
        rf = rf.reindex(y.index)

        if rf_is_annualized:
            rf = annual_to_period_return(rf, horizon_days=horizon_days, periods_per_year=periods_per_year)

        return y - rf

    rf = float(risk_free_rate)
    if rf_is_annualized:
        rf = annual_to_period_return(rf, horizon_days=horizon_days, periods_per_year=periods_per_year)

    return y - rf


def add_excess_target_to_panel(
    panel_df: pd.DataFrame,
    target_col: str = "target_5d",
    output_col: str = "target_5d_excess",
    risk_free_rate: float | pd.Series | None = None,
    horizon_days: int = 5,
    periods_per_year: int = 252,
    rf_is_annualized: bool = True,
) -> pd.DataFrame:
    """
    panel 데이터에 excess return target 컬럼 추가
    """
    if target_col not in panel_df.columns:
        raise ValueError(f"{target_col} 컬럼이 없습니다.")

    out = panel_df.copy()
    out[output_col] = make_excess_return_target_from_series(
        out[target_col],
        risk_free_rate=risk_free_rate,
        horizon_days=horizon_days,
        periods_per_year=periods_per_year,
        rf_is_annualized=rf_is_annualized,
    )
    return out


def standardize_target_column_name(
    panel_df: pd.DataFrame,
    source_col: str,
    target_col: str = "target",
) -> pd.DataFrame:
    """
    panel 데이터에서 특정 타깃 컬럼명을 표준화
    """
    if source_col not in panel_df.columns:
        raise ValueError(f"{source_col} 컬럼이 없습니다.")

    out = panel_df.copy()
    out = out.rename(columns={source_col: target_col})
    return out


def build_target_table_from_panel(
    panel_df: pd.DataFrame,
    target_col: str = "target_5d",
) -> pd.DataFrame:
    """
    panel(long) -> date x asset wide target table
    """
    required = {"date", "asset", target_col}
    missing = required - set(panel_df.columns)
    if missing:
        raise ValueError(f"필수 컬럼이 없습니다: {missing}")

    out = panel_df.pivot(index="date", columns="asset", values=target_col)
    out.index = pd.to_datetime(out.index)
    out = out.sort_index().sort_index(axis=1)

    return out