from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass

import pandas as pd

from src.backtest.engine import BacktestEngine, BacktestResult
from src.backtest.performance import (
    compute_cumulative_wealth,
    make_performance_table,
)


@dataclass
class ModelComparisonResult:
    summary: pd.DataFrame
    returns: pd.DataFrame
    wealth: pd.DataFrame
    raw_results: dict[str, BacktestResult]
    benchmark_name: str


def run_expected_return_model_comparison(
    prices: pd.DataFrame,
    return_models: dict[str, object],
    risk_model,
    allocation_model,
    lookback_periods: int = 36,
    transaction_cost: float = 0.001,
    resample_freq: str = "ME",
    periods_per_year: int = 12,
    risk_free_rate: float = 0.0,
    charge_initial_cost: bool = False,
    benchmark_name: str = "EqualWeight",
    include_benchmark: bool = True,
    debug: bool = False,
    debug_max_steps: int | None = None,
) -> ModelComparisonResult:
    """
    같은 risk model / allocation model / backtest engine 위에서
    기대수익률 모델들만 바꿔가며 성과 비교.
    """
    if not isinstance(return_models, dict) or len(return_models) == 0:
        raise ValueError("return_models는 비어 있지 않은 dict여야 합니다.")

    raw_results: dict[str, BacktestResult] = {}
    returns_map: dict[str, pd.Series] = {}
    turnover_map: dict[str, pd.Series] = {}

    benchmark_returns: pd.Series | None = None
    benchmark_turnover: pd.Series | None = None

    reference_universe: tuple[str, ...] | None = None

    for model_name, return_model in return_models.items():
        engine = BacktestEngine(
            return_model=deepcopy(return_model),
            risk_model=deepcopy(risk_model),
            allocation_model=deepcopy(allocation_model),
            lookback_periods=lookback_periods,
            transaction_cost=transaction_cost,
            resample_freq=resample_freq,
            periods_per_year=periods_per_year,
            risk_free_rate=risk_free_rate,
            charge_initial_cost=charge_initial_cost,
            debug=debug,
            debug_max_steps=debug_max_steps,
        )

        bt = engine.run(prices)

        current_universe = tuple(bt.weights_history.columns.tolist())
        if reference_universe is None:
            reference_universe = current_universe
        elif current_universe != reference_universe:
            raise ValueError(
                "모델별로 최적화 자산 유니버스가 다릅니다. "
                "같은 tickers, 같은 전처리, 같은 min_obs 조건을 맞춰 주세요.\n"
                f"기준 유니버스: {reference_universe}\n"
                f"{model_name} 유니버스: {current_universe}"
            )

        raw_results[model_name] = bt
        returns_map[model_name] = bt.portfolio_returns.rename(model_name)
        turnover_map[model_name] = bt.portfolio_turnover.rename(model_name)

        if benchmark_returns is None:
            benchmark_returns = bt.benchmark_returns.rename(benchmark_name)
        if benchmark_turnover is None:
            benchmark_turnover = bt.benchmark_turnover.rename(benchmark_name)

    if include_benchmark and benchmark_returns is not None and benchmark_turnover is not None:
        returns_map[benchmark_name] = benchmark_returns
        turnover_map[benchmark_name] = benchmark_turnover

    summary = make_performance_table(
        returns_map=returns_map,
        turnover_map=turnover_map,
        periods_per_year=periods_per_year,
        risk_free_rate=risk_free_rate,
    )

    returns_df = pd.concat(returns_map, axis=1).sort_index()

    wealth_map = {
        name: compute_cumulative_wealth(series)
        for name, series in returns_map.items()
    }
    wealth_df = pd.concat(wealth_map, axis=1).sort_index()

    return ModelComparisonResult(
        summary=summary,
        returns=returns_df,
        wealth=wealth_df,
        raw_results=raw_results,
        benchmark_name=benchmark_name,
    )