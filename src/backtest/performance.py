from __future__ import annotations

from typing import Mapping

import numpy as np
import pandas as pd


def compute_cumulative_wealth(
    returns: pd.Series,
    start_value: float = 1.0,
) -> pd.Series:
    """
    수익률 시계열 -> 누적 자산가치
    """
    r = returns.dropna().copy()
    wealth = start_value * (1.0 + r).cumprod()
    wealth.name = "wealth"
    return wealth


def compute_drawdown(returns: pd.Series) -> pd.Series:
    """
    수익률 시계열 -> drawdown 시계열
    """
    wealth = compute_cumulative_wealth(returns)
    running_max = wealth.cummax()
    drawdown = wealth / running_max - 1.0
    drawdown.name = "drawdown"
    return drawdown


def total_return(returns: pd.Series) -> float:
    r = returns.dropna()
    if len(r) == 0:
        return np.nan
    return float((1.0 + r).prod() - 1.0)


def annualized_return(
    returns: pd.Series,
    periods_per_year: int = 12,
) -> float:
    """
    기하평균 기반 연율 수익률
    """
    r = returns.dropna()
    if len(r) == 0:
        return np.nan

    growth = float((1.0 + r).prod())
    n_periods = len(r)

    return float(growth ** (periods_per_year / n_periods) - 1.0)


def annualized_volatility(
    returns: pd.Series,
    periods_per_year: int = 12,
) -> float:
    r = returns.dropna()
    if len(r) < 2:
        return np.nan
    return float(r.std(ddof=1) * np.sqrt(periods_per_year))


def sharpe_ratio(
    returns: pd.Series,
    risk_free_rate: float = 0.0,
    periods_per_year: int = 12,
) -> float:
    """
    연율 무위험수익률을 받아 샤프비율 계산
    """
    r = returns.dropna()
    if len(r) < 2:
        return np.nan

    rf_per_period = (1.0 + risk_free_rate) ** (1.0 / periods_per_year) - 1.0
    excess = r - rf_per_period

    vol = excess.std(ddof=1)
    if np.isclose(vol, 0.0):
        return np.nan

    return float(excess.mean() / vol * np.sqrt(periods_per_year))


def max_drawdown(returns: pd.Series) -> float:
    dd = compute_drawdown(returns)
    if dd.empty:
        return np.nan
    return float(dd.min())


def summarize_backtest(
    returns: pd.Series,
    turnover: pd.Series | None = None,
    periods_per_year: int = 12,
    risk_free_rate: float = 0.0,
) -> pd.Series:
    """
    단일 전략의 성과 요약
    """
    r = returns.dropna()

    summary = pd.Series(
        {
            "n_periods": len(r),
            "total_return": total_return(r),
            "annualized_return": annualized_return(r, periods_per_year=periods_per_year),
            "annualized_volatility": annualized_volatility(r, periods_per_year=periods_per_year),
            "sharpe_ratio": sharpe_ratio(
                r,
                risk_free_rate=risk_free_rate,
                periods_per_year=periods_per_year,
            ),
            "max_drawdown": max_drawdown(r),
        }
    )

    if turnover is not None:
        t = turnover.dropna()
        summary["average_turnover"] = float(t.mean()) if len(t) > 0 else np.nan
        summary["annualized_turnover"] = (
            float(t.mean()) * periods_per_year if len(t) > 0 else np.nan
        )

    return summary


def make_performance_table(
    returns_map: Mapping[str, pd.Series],
    turnover_map: Mapping[str, pd.Series] | None = None,
    periods_per_year: int = 12,
    risk_free_rate: float = 0.0,
) -> pd.DataFrame:
    """
    여러 전략을 한 표로 요약
    """
    rows = {}
    for name, returns in returns_map.items():
        turnover = None
        if turnover_map is not None and name in turnover_map:
            turnover = turnover_map[name]

        rows[name] = summarize_backtest(
            returns=returns,
            turnover=turnover,
            periods_per_year=periods_per_year,
            risk_free_rate=risk_free_rate,
        )

    table = pd.DataFrame(rows).T
    return table