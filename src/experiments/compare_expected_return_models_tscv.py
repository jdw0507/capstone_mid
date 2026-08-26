from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.data.preprocessing import (
    prepare_price_data_for_allocation,
    subset_by_date,
)
from src.backtest.engine import BacktestEngine
from src.experiments.compare_expected_return_models import (
    run_expected_return_model_comparison,
)


@dataclass
class WalkForwardFoldResult:
    fold_id: int
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    summary: pd.DataFrame
    wealth: pd.DataFrame
    returns: pd.DataFrame


@dataclass
class WalkForwardComparisonResult:
    fold_results: list[WalkForwardFoldResult]
    fold_summary_long: pd.DataFrame
    aggregate_summary: pd.DataFrame
    per_fold_metric_table: pd.DataFrame


def build_walk_forward_splits(
    monthly_prices: pd.DataFrame,
    train_window_months: int = 84,
    test_window_months: int = 36,
    step_months: int = 6,
) -> pd.DataFrame:
    """
    월말 가격 데이터 기준 walk-forward split 생성

    예:
    - train 84개월
    - test 36개월
    - step 6개월
    """
    prices = monthly_prices.copy()
    prices.index = pd.to_datetime(prices.index)
    prices = prices.sort_index()
    prices = prices[~prices.index.duplicated(keep="first")]

    n = len(prices)
    total_needed = train_window_months + test_window_months
    if n < total_needed:
        raise ValueError(
            f"데이터가 부족합니다. 최소 {total_needed}개월 필요, 현재 {n}개월."
        )

    dates = prices.index
    rows = []
    fold_id = 0

    start_idx = 0
    while True:
        train_start_idx = start_idx
        train_end_idx = train_start_idx + train_window_months - 1
        test_start_idx = train_end_idx + 1
        test_end_idx = test_start_idx + test_window_months - 1

        if test_end_idx >= n:
            break

        rows.append(
            {
                "fold_id": fold_id,
                "train_start": dates[train_start_idx],
                "train_end": dates[train_end_idx],
                "test_start": dates[test_start_idx],
                "test_end": dates[test_end_idx],
            }
        )

        fold_id += 1
        start_idx += step_months

    if len(rows) == 0:
        raise ValueError("생성된 walk-forward fold가 없습니다.")

    return pd.DataFrame(rows)


def run_expected_return_model_comparison_walk_forward(
    prices: pd.DataFrame,
    return_models: dict[str, object],
    risk_model,
    allocation_model,
    risk_parity_allocator=None,
    dummy_return_model_for_rp=None,
    train_window_months: int = 84,
    test_window_months: int = 36,
    step_months: int = 6,
    resample_freq: str = "ME",
    periods_per_year: int = 12,
    risk_free_rate: float = 0.0,
    transaction_cost: float = 0.001,
    charge_initial_cost: bool = False,
    include_risk_parity: bool = True,
    benchmark_name: str = "EqualWeight",
    debug: bool = False,
    debug_max_steps: int | None = None,
) -> WalkForwardComparisonResult:
    """
    Walk-forward 방식으로 여러 기대수익률 모델 비교.
    """

    # 전체 데이터 기준 월말 가격 생성
    monthly_prices_full = prepare_price_data_for_allocation(
        prices=prices,
        use_common_period=True,
        resample_freq=resample_freq,
    )

    split_df = build_walk_forward_splits(
        monthly_prices=monthly_prices_full,
        train_window_months=train_window_months,
        test_window_months=test_window_months,
        step_months=step_months,
    )

    fold_results: list[WalkForwardFoldResult] = []
    fold_summary_rows = []

    for row in split_df.itertuples(index=False):
        fold_id = int(row.fold_id)
        train_start = pd.Timestamp(row.train_start)
        train_end = pd.Timestamp(row.train_end)
        test_start = pd.Timestamp(row.test_start)
        test_end = pd.Timestamp(row.test_end)

        # BacktestEngine는 slice 전체에서:
        # train_window_months를 lookback으로 보고,
        # 그 이후 월별 리밸런싱 결과를 계산하므로
        # fold 구간은 train_start ~ test_end로 자르면 됨.
        fold_prices = subset_by_date(
            prices,
            start=train_start.strftime("%Y-%m-%d"),
            end=test_end.strftime("%Y-%m-%d"),
        )

        mvo_comp = run_expected_return_model_comparison(
            prices=fold_prices,
            return_models=deepcopy(return_models),
            risk_model=deepcopy(risk_model),
            allocation_model=deepcopy(allocation_model),
            lookback_periods=train_window_months,
            transaction_cost=transaction_cost,
            resample_freq=resample_freq,
            periods_per_year=periods_per_year,
            risk_free_rate=risk_free_rate,
            charge_initial_cost=charge_initial_cost,
            benchmark_name=benchmark_name,
            include_benchmark=True,
            debug=debug,
            debug_max_steps=debug_max_steps,
        )

        final_summary = mvo_comp.summary.copy()
        final_wealth = mvo_comp.wealth.copy()
        final_returns = mvo_comp.returns.copy()

        if include_risk_parity and risk_parity_allocator is not None:
            if dummy_return_model_for_rp is None:
                raise ValueError("RiskParity를 포함하려면 dummy_return_model_for_rp를 넣어주세요.")

            rp_backtester = BacktestEngine(
                return_model=deepcopy(dummy_return_model_for_rp),
                risk_model=deepcopy(risk_model),
                allocation_model=deepcopy(risk_parity_allocator),
                lookback_periods=train_window_months,
                transaction_cost=transaction_cost,
                resample_freq=resample_freq,
                periods_per_year=periods_per_year,
                risk_free_rate=risk_free_rate,
                charge_initial_cost=charge_initial_cost,
                debug=debug,
                debug_max_steps=debug_max_steps,
            )
            # RiskParity는 expected return을 실제로 사용하지 않으므로
            # return_model은 첫 번째 모델과 동일 인터페이스만 맞추면 됨.
            # 다만 clone 안정성을 위해 아래처럼 HistoricalMean을 쓰는 것도 가능.

            rp_result = rp_backtester.run(fold_prices)

            rp_summary = rp_result.summary.loc[["mvo_strategy"]].copy()
            rp_summary.index = ["RiskParity"]
            final_summary = pd.concat([final_summary, rp_summary], axis=0)

            final_wealth["RiskParity"] = (1.0 + rp_result.portfolio_returns).cumprod()
            final_returns["RiskParity"] = rp_result.portfolio_returns

        desired_order = [benchmark_name] + list(return_models.keys())
        if include_risk_parity and risk_parity_allocator is not None:
            desired_order += ["RiskParity"]

        final_summary = final_summary.reindex(
            [idx for idx in desired_order if idx in final_summary.index]
        )

        final_summary = final_summary[
            [
                "n_periods",
                "total_return",
                "annualized_return",
                "annualized_volatility",
                "sharpe_ratio",
                "max_drawdown",
                "average_turnover",
                "annualized_turnover",
            ]
        ]

        wealth_cols = [c for c in desired_order if c in final_wealth.columns]
        return_cols = [c for c in desired_order if c in final_returns.columns]

        final_wealth = final_wealth[wealth_cols]
        final_returns = final_returns[return_cols]

        fold_result = WalkForwardFoldResult(
            fold_id=fold_id,
            train_start=train_start,
            train_end=train_end,
            test_start=test_start,
            test_end=test_end,
            summary=final_summary,
            wealth=final_wealth,
            returns=final_returns,
        )
        fold_results.append(fold_result)

        tmp = final_summary.copy()
        tmp["fold_id"] = fold_id
        tmp["train_start"] = train_start
        tmp["train_end"] = train_end
        tmp["test_start"] = test_start
        tmp["test_end"] = test_end
        tmp["model"] = tmp.index
        fold_summary_rows.append(tmp.reset_index(drop=True))

    fold_summary_long = pd.concat(fold_summary_rows, axis=0, ignore_index=True)

    metric_cols = [
        "total_return",
        "annualized_return",
        "annualized_volatility",
        "sharpe_ratio",
        "max_drawdown",
        "average_turnover",
        "annualized_turnover",
    ]

    grouped = fold_summary_long.groupby("model")

    aggregate_mean = grouped[metric_cols].mean().add_prefix("mean_")
    aggregate_std = grouped[metric_cols].std().add_prefix("std_")
    aggregate_median = grouped[metric_cols].median().add_prefix("median_")

    # benchmark 승률
    benchmark_df = fold_summary_long[fold_summary_long["model"] == benchmark_name][
        ["fold_id", "annualized_return", "sharpe_ratio"]
    ].rename(
        columns={
            "annualized_return": "benchmark_ann_return",
            "sharpe_ratio": "benchmark_sharpe",
        }
    )

    merged = fold_summary_long.merge(benchmark_df, on="fold_id", how="left")
    merged["win_vs_benchmark_return"] = (
        merged["annualized_return"] > merged["benchmark_ann_return"]
    ).astype(float)
    merged["win_vs_benchmark_sharpe"] = (
        merged["sharpe_ratio"] > merged["benchmark_sharpe"]
    ).astype(float)

    win_rate = merged.groupby("model")[
        ["win_vs_benchmark_return", "win_vs_benchmark_sharpe"]
    ].mean()

    aggregate_summary = pd.concat(
        [aggregate_mean, aggregate_std, aggregate_median, win_rate],
        axis=1,
    ).sort_values("mean_sharpe_ratio", ascending=False)

    per_fold_metric_table = fold_summary_long.pivot_table(
        index="fold_id",
        columns="model",
        values="sharpe_ratio",
    )

    return WalkForwardComparisonResult(
        fold_results=fold_results,
        fold_summary_long=fold_summary_long,
        aggregate_summary=aggregate_summary,
        per_fold_metric_table=per_fold_metric_table,
    )


def _safe_clone_kwargs(model) -> dict:
    """
    아주 제한적인 fallback용.
    네 프로젝트에선 RiskParity용 dummy return_model로 HistoricalMean을 직접 넣는 게 더 명확함.
    """
    out = {}
    for k, v in getattr(model, "__dict__", {}).items():
        if k.startswith("last_"):
            continue
        out[k] = v
    return out