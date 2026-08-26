from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

import numpy as np
import pandas as pd

from copy import deepcopy
from src.expected_returns.black_litterman import BlackLittermanExpectedReturn
from src.backtest.performance import summarize_backtest, compute_cumulative_wealth
from src.data.splitters import build_prediction_schedule
from src.views.q_builder import (
    PredictionSnapshotEmptyError,
    get_prediction_snapshot,
    build_absolute_q,
    build_relative_q,
)
from src.views.p_builder import (
    build_absolute_p,
    build_relative_p,
    combine_p_matrices,
)
from src.views.omega_builder import (
    build_absolute_omega_from_uncertainty,
    build_relative_omega_from_uncertainty,
    build_asset_error_statistics,
    build_absolute_omega_from_error_history,
    build_relative_omega_from_error_history,
    combine_omega_matrices,
)


ViewMode = Literal["no_view", "absolute", "relative", "mixed"]
OmegaMode = Literal["uncertainty", "error_history"]


@dataclass
class StaticBLStrategyConfig:
    name: str
    mode: ViewMode = "no_view"

    absolute_model_name: Optional[str] = None
    relative_model_name: Optional[str] = None

    absolute_top_k: int = 3
    absolute_bottom_k: int = 0
    relative_top_k: int = 2
    relative_bottom_k: int = 2
    relative_pairing: Literal["zip", "cartesian"] = "zip"

    annualize_q: bool = False
    horizon_days: int = 5
    periods_per_year: int = 252
    min_abs_pred: Optional[float] = None

    omega_method: OmegaMode = "uncertainty"
    annualize_omega: bool = False
    error_metric: str = "rmse"

    omega_scale: float = 1.0
    fallback_to_no_view: bool = True


@dataclass
class StaticBLFoldResult:
    fold_id: int
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    strategy_summary: pd.DataFrame
    strategy_returns: pd.DataFrame
    strategy_wealth: pd.DataFrame
    step_results: pd.DataFrame


@dataclass
class StaticBLRunResult:
    fold_results: list[StaticBLFoldResult]
    step_results: pd.DataFrame
    fold_summary_long: pd.DataFrame
    aggregate_summary: pd.DataFrame


def extract_close_price_table_from_wide_df(
    df: pd.DataFrame,
    assets: Optional[list[str]] = None,
    date_col: str = "Date",
    close_prefix: str = "Close_",
) -> pd.DataFrame:
    if date_col not in df.columns:
        raise ValueError(f"{date_col} column not found.")

    out = df.copy()
    out[date_col] = pd.to_datetime(out[date_col])
    out = out.sort_values(date_col).drop_duplicates(subset=[date_col], keep="first")

    close_cols = [c for c in out.columns if c.startswith(close_prefix)]
    if len(close_cols) == 0:
        raise ValueError(f"No columns with prefix {close_prefix} found.")

    px = out[[date_col] + close_cols].copy().set_index(date_col)
    rename_map = {c: c.replace(close_prefix, "") for c in close_cols}
    px = px.rename(columns=rename_map)

    if assets is not None:
        keep = [a for a in assets if a in px.columns]
        if len(keep) == 0:
            raise ValueError("None of the requested assets exist in close price table.")
        px = px[keep]

    px = px.apply(pd.to_numeric, errors="coerce")
    px = px.sort_index().sort_index(axis=1)
    return px


def _compute_turnover(
    current_weights: pd.Series | None,
    target_weights: pd.Series,
    charge_initial_cost: bool = False,
) -> float:
    target = target_weights.sort_index()

    if current_weights is None:
        return 1.0 if charge_initial_cost else 0.0

    current = current_weights.reindex(target.index).fillna(0.0).sort_index()
    turnover = 0.5 * float(np.abs(target - current).sum())
    return turnover


def _drift_weights(
    target_weights: pd.Series,
    realized_returns: pd.Series,
) -> pd.Series:
    target = target_weights.sort_index()
    realized = realized_returns.reindex(target.index).astype(float).sort_index()

    end_values = target * (1.0 + realized)
    total_value = float(end_values.sum())

    if total_value <= 0:
        raise ValueError("Portfolio value became non-positive during drift step.")

    return (end_values / total_value).rename("weight")


def _slice_history_returns(
    close_prices: pd.DataFrame,
    end_date: pd.Timestamp,
    assets: list[str],
    lookback_days: Optional[int] = None,
) -> pd.DataFrame:
    hist_px = close_prices.loc[:end_date, assets].copy()
    hist_ret = hist_px.pct_change().dropna(how="all")

    if lookback_days is not None and len(hist_ret) > lookback_days:
        hist_ret = hist_ret.tail(lookback_days)

    return hist_ret


def _realized_holding_return(
    close_prices: pd.DataFrame,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    assets: list[str],
) -> pd.Series:
    start_px = close_prices.loc[start_date, assets]
    end_px = close_prices.loc[end_date, assets]
    r = end_px / start_px - 1.0
    return r.astype(float).rename("realized_return")


def _build_views_for_strategy(
    strategy: StaticBLStrategyConfig,
    assets: list[str],
    decision_date: pd.Timestamp,
    fold_id: int,
    per_prediction_test: pd.DataFrame,
    per_prediction_view: Optional[pd.DataFrame] = None,
) -> tuple[pd.Series | None, pd.DataFrame | None, pd.DataFrame | None, pd.DataFrame]:
    empty_meta = pd.DataFrame()

    if strategy.mode == "no_view":
        return None, None, None, empty_meta

    try:
        if strategy.mode == "absolute":
            if strategy.absolute_model_name is None:
                raise ValueError("absolute_model_name is required for absolute view.")

            snap = get_prediction_snapshot(
                per_prediction=per_prediction_test,
                model_name=strategy.absolute_model_name,
                date=decision_date,
                fold_id=fold_id,
                evaluation_split="test",
            )

            Q_abs, meta_abs = build_absolute_q(
                snapshot=snap,
                top_k=strategy.absolute_top_k,
                bottom_k=strategy.absolute_bottom_k,
                min_abs_pred=strategy.min_abs_pred,
                annualize=strategy.annualize_q,
                horizon_days=strategy.horizon_days,
                periods_per_year=strategy.periods_per_year,
                allow_negative_absolute_views=True,
            )

            P_abs = build_absolute_p(assets=assets, meta_abs=meta_abs)

            if strategy.omega_method == "uncertainty":
                Omega_abs = build_absolute_omega_from_uncertainty(
                    meta_abs=meta_abs,
                    annualize=strategy.annualize_omega,
                    horizon_days=strategy.horizon_days,
                    periods_per_year=strategy.periods_per_year,
                )
            else:
                if per_prediction_view is None:
                    raise ValueError("per_prediction_view is required for error_history omega.")
                stats = build_asset_error_statistics(
                    per_prediction=per_prediction_view,
                    model_name=strategy.absolute_model_name,
                    date=decision_date,
                    fold_id=fold_id,
                    evaluation_split="view_build",
                    window_days=252,
                    min_obs=20,
                )
                Omega_abs = build_absolute_omega_from_error_history(
                    meta_abs=meta_abs,
                    asset_error_stats=stats,
                    annualize=strategy.annualize_omega,
                    horizon_days=strategy.horizon_days,
                    periods_per_year=strategy.periods_per_year,
                    error_metric=strategy.error_metric,
                )

            Omega_abs = strategy.omega_scale * Omega_abs
            return Q_abs, P_abs, Omega_abs, meta_abs

        elif strategy.mode == "relative":
            if strategy.relative_model_name is None:
                raise ValueError("relative_model_name is required for relative view.")

            snap = get_prediction_snapshot(
                per_prediction=per_prediction_test,
                model_name=strategy.relative_model_name,
                date=decision_date,
                fold_id=fold_id,
                evaluation_split="test",
            )

            Q_rel, meta_rel = build_relative_q(
                snapshot=snap,
                top_k=strategy.relative_top_k,
                bottom_k=strategy.relative_bottom_k,
                pairing=strategy.relative_pairing,
                annualize=strategy.annualize_q,
                horizon_days=strategy.horizon_days,
                periods_per_year=strategy.periods_per_year,
            )

            P_rel = build_relative_p(assets=assets, meta_rel=meta_rel)

            if strategy.omega_method == "uncertainty":
                Omega_rel = build_relative_omega_from_uncertainty(
                    meta_rel=meta_rel,
                    annualize=strategy.annualize_omega,
                    horizon_days=strategy.horizon_days,
                    periods_per_year=strategy.periods_per_year,
                )
            else:
                if per_prediction_view is None:
                    raise ValueError("per_prediction_view is required for error_history omega.")
                stats = build_asset_error_statistics(
                    per_prediction=per_prediction_view,
                    model_name=strategy.relative_model_name,
                    date=decision_date,
                    fold_id=fold_id,
                    evaluation_split="view_build",
                    window_days=252,
                    min_obs=20,
                )
                Omega_rel = build_relative_omega_from_error_history(
                    meta_rel=meta_rel,
                    asset_error_stats=stats,
                    annualize=strategy.annualize_omega,
                    horizon_days=strategy.horizon_days,
                    periods_per_year=strategy.periods_per_year,
                    error_metric=strategy.error_metric,
                )

            Omega_rel = strategy.omega_scale * Omega_rel
            return Q_rel, P_rel, Omega_rel, meta_rel

        elif strategy.mode == "mixed":
            if strategy.absolute_model_name is None or strategy.relative_model_name is None:
                raise ValueError("mixed view requires both absolute_model_name and relative_model_name.")

            snap_abs = get_prediction_snapshot(
                per_prediction=per_prediction_test,
                model_name=strategy.absolute_model_name,
                date=decision_date,
                fold_id=fold_id,
                evaluation_split="test",
            )
            snap_rel = get_prediction_snapshot(
                per_prediction=per_prediction_test,
                model_name=strategy.relative_model_name,
                date=decision_date,
                fold_id=fold_id,
                evaluation_split="test",
            )

            Q_abs, meta_abs = build_absolute_q(
                snapshot=snap_abs,
                top_k=strategy.absolute_top_k,
                bottom_k=strategy.absolute_bottom_k,
                min_abs_pred=strategy.min_abs_pred,
                annualize=strategy.annualize_q,
                horizon_days=strategy.horizon_days,
                periods_per_year=strategy.periods_per_year,
                allow_negative_absolute_views=True,
            )
            P_abs = build_absolute_p(assets=assets, meta_abs=meta_abs)

            Q_rel, meta_rel = build_relative_q(
                snapshot=snap_rel,
                top_k=strategy.relative_top_k,
                bottom_k=strategy.relative_bottom_k,
                pairing=strategy.relative_pairing,
                annualize=strategy.annualize_q,
                horizon_days=strategy.horizon_days,
                periods_per_year=strategy.periods_per_year,
            )
            P_rel = build_relative_p(assets=assets, meta_rel=meta_rel)

            if strategy.omega_method == "uncertainty":
                Omega_abs = build_absolute_omega_from_uncertainty(
                    meta_abs=meta_abs,
                    annualize=strategy.annualize_omega,
                    horizon_days=strategy.horizon_days,
                    periods_per_year=strategy.periods_per_year,
                )
                Omega_rel = build_relative_omega_from_uncertainty(
                    meta_rel=meta_rel,
                    annualize=strategy.annualize_omega,
                    horizon_days=strategy.horizon_days,
                    periods_per_year=strategy.periods_per_year,
                )
            else:
                if per_prediction_view is None:
                    raise ValueError("per_prediction_view is required for error_history omega.")

                stats_abs = build_asset_error_statistics(
                    per_prediction=per_prediction_view,
                    model_name=strategy.absolute_model_name,
                    date=decision_date,
                    fold_id=fold_id,
                    evaluation_split="view_build",
                    window_days=252,
                    min_obs=20,
                )
                stats_rel = build_asset_error_statistics(
                    per_prediction=per_prediction_view,
                    model_name=strategy.relative_model_name,
                    date=decision_date,
                    fold_id=fold_id,
                    evaluation_split="view_build",
                    window_days=252,
                    min_obs=20,
                )

                Omega_abs = build_absolute_omega_from_error_history(
                    meta_abs=meta_abs,
                    asset_error_stats=stats_abs,
                    annualize=strategy.annualize_omega,
                    horizon_days=strategy.horizon_days,
                    periods_per_year=strategy.periods_per_year,
                    error_metric=strategy.error_metric,
                )
                Omega_rel = build_relative_omega_from_error_history(
                    meta_rel=meta_rel,
                    asset_error_stats=stats_rel,
                    annualize=strategy.annualize_omega,
                    horizon_days=strategy.horizon_days,
                    periods_per_year=strategy.periods_per_year,
                    error_metric=strategy.error_metric,
                )

            Omega_abs = strategy.omega_scale * Omega_abs
            Omega_rel = strategy.omega_scale * Omega_rel

            Q = pd.concat([Q_abs, Q_rel], axis=0)
            P = combine_p_matrices(P_abs, P_rel)
            Omega = combine_omega_matrices(Omega_abs, Omega_rel)
            meta = pd.concat([meta_abs, meta_rel], axis=0, ignore_index=True)

            return Q, P, Omega, meta

        else:
            raise ValueError(f"Unsupported strategy.mode: {strategy.mode}")

    except PredictionSnapshotEmptyError as e:
        meta = pd.DataFrame([{"fallback_reason": str(e), "fallback_type": "missing_snapshot"}])
        return None, None, None, meta
    except Exception as e:
        if strategy.fallback_to_no_view:
            meta = pd.DataFrame([{"fallback_reason": str(e)}])
            return None, None, None, meta
        raise


def run_static_bl_view_strategies(
    close_prices: pd.DataFrame,
    splits: pd.DataFrame,
    per_prediction_test: pd.DataFrame,
    strategies: list[StaticBLStrategyConfig],
    risk_model,
    allocation_model,
    per_prediction_view: Optional[pd.DataFrame] = None,
    bl_lookback_days: Optional[int] = None,
    rebalance_every_n_days: int = 5,
    forecast_horizon_days: int = 5,
    periods_per_year: int = 252,
    risk_free_rate: float = 0.02,
    market_ticker: str = "SPY",
    market_weights: Optional[pd.Series] = None,
    market_caps: Optional[pd.Series] = None,
    tau: float = 0.05,
    risk_aversion: float = 2.5,
    transaction_cost: float = 0.0,
    charge_initial_cost: bool = False,
    benchmark_strategy_name: str = "BL_NoView",
    verbose_progress: bool = True,
) -> StaticBLRunResult:
    close_prices = close_prices.copy()
    close_prices.index = pd.to_datetime(close_prices.index)
    close_prices = close_prices.sort_index()
    assets = close_prices.columns.tolist()

    fold_results: list[StaticBLFoldResult] = []
    step_rows = []
    fold_summary_rows = []

    for split_idx, split_row in enumerate(splits.itertuples(index=False), start=1):
        fold_id = int(split_row.fold_id)
        test_start = pd.Timestamp(split_row.test_start)
        test_end = pd.Timestamp(split_row.test_end)

        if verbose_progress:
            print(
                f"\n=== Static BL Fold {split_idx}/{len(splits)} | fold_id={fold_id} "
                f"| test: {test_start.date()} ~ {test_end.date()} ==="
            )

        schedule = build_prediction_schedule(
            close_prices,
            start_date=test_start,
            end_date=test_end,
            every_n_days=rebalance_every_n_days,
            forecast_horizon_days=forecast_horizon_days,
            date_col="date",
        )

        strategy_returns = {}
        strategy_turnover = {}
        strategy_step_frames = []

        for strategy_idx, strategy in enumerate(strategies, start=1):
            if verbose_progress:
                print(f"[fold {fold_id}] Strategy {strategy_idx}/{len(strategies)}: {strategy.name}")

            current_weights = None
            net_return_map = {}
            turnover_map = {}
            step_records = []

            for step in schedule.itertuples(index=False):
                decision_date = pd.Timestamp(step.decision_date)
                forecast_end_date = pd.Timestamp(step.forecast_end_date)

                hist_returns = _slice_history_returns(
                    close_prices=close_prices,
                    end_date=decision_date,
                    assets=assets,
                    lookback_days=bl_lookback_days,
                )

                if hist_returns.empty:
                    continue

                Q, P, Omega, meta_views = _build_views_for_strategy(
                    strategy=strategy,
                    assets=assets,
                    decision_date=decision_date,
                    fold_id=fold_id,
                    per_prediction_test=per_prediction_test,
                    per_prediction_view=per_prediction_view,
                )

                bl_model = BlackLittermanExpectedReturn(
                    tickers=assets,
                    prices=None,
                    periods_per_year=periods_per_year,
                    min_obs=min(252, max(20, len(hist_returns))),
                    tau=tau,
                    risk_free_rate=risk_free_rate,
                    market_ticker=market_ticker,
                    market_prices=None,
                    market_caps=market_caps,
                    market_weights=market_weights,
                    risk_aversion=risk_aversion,
                    P=P,
                    Q=Q,
                    Omega=Omega,
                    view_confidences=None,
                    use_market_cap_weights=True,
                    allow_equal_weight_fallback=True,
                    market_weight_fallback="equal_weight",
                    blend_with_equal_weight=None,
                )

                mu = bl_model.fit_predict(hist_returns[assets])

                cov_model = deepcopy(risk_model)
                cov = cov_model.fit_predict(hist_returns[mu.index])

                allocator = deepcopy(allocation_model)
                weights = allocator.optimize(mu, cov).sort_index()

                realized_asset_return = _realized_holding_return(
                    close_prices=close_prices,
                    start_date=decision_date,
                    end_date=forecast_end_date,
                    assets=list(weights.index),
                )

                gross_return = float(weights @ realized_asset_return)
                turnover = _compute_turnover(
                    current_weights=current_weights,
                    target_weights=weights,
                    charge_initial_cost=charge_initial_cost,
                )
                net_return = gross_return - turnover * transaction_cost

                net_return_map[forecast_end_date] = net_return
                turnover_map[forecast_end_date] = turnover

                current_weights = _drift_weights(weights, realized_asset_return)

                step_records.append(
                    {
                        "fold_id": fold_id,
                        "strategy": strategy.name,
                        "decision_date": decision_date,
                        "forecast_end_date": forecast_end_date,
                        "gross_return": gross_return,
                        "net_return": net_return,
                        "turnover": turnover,
                        "n_views": 0 if Q is None else len(Q),
                        "used_no_view": int(Q is None),
                        "mean_q": np.nan if Q is None else float(Q.mean()),
                        "abs_mean_q": np.nan if Q is None else float(Q.abs().mean()),
                    }
                )

            r = pd.Series(net_return_map, name=strategy.name).sort_index()
            t = pd.Series(turnover_map, name=strategy.name).sort_index()

            strategy_returns[strategy.name] = r
            strategy_step_frames.append(pd.DataFrame(step_records))

            summary = summarize_backtest(
                returns=r,
                turnover=t,
                periods_per_year=periods_per_year / forecast_horizon_days,
                risk_free_rate=risk_free_rate,
            )
            summary["fold_id"] = fold_id
            summary["strategy"] = strategy.name
            summary["test_start"] = test_start
            summary["test_end"] = test_end
            fold_summary_rows.append(summary)

        strategy_returns_df = pd.concat(strategy_returns, axis=1).sort_index()
        strategy_wealth_df = pd.concat(
            {k: compute_cumulative_wealth(v) for k, v in strategy_returns.items()},
            axis=1,
        ).sort_index()

        strategy_summary_df = (
            pd.DataFrame(fold_summary_rows)
            .query("fold_id == @fold_id")
            .set_index("strategy")
            .sort_index()
        )

        fold_step_results = pd.concat(strategy_step_frames, axis=0, ignore_index=True) if strategy_step_frames else pd.DataFrame()

        fold_results.append(
            StaticBLFoldResult(
                fold_id=fold_id,
                test_start=test_start,
                test_end=test_end,
                strategy_summary=strategy_summary_df,
                strategy_returns=strategy_returns_df,
                strategy_wealth=strategy_wealth_df,
                step_results=fold_step_results,
            )
        )

        if not fold_step_results.empty:
            step_rows.append(fold_step_results)

    step_results = pd.concat(step_rows, axis=0, ignore_index=True) if step_rows else pd.DataFrame()

    fold_summary_long = pd.DataFrame(fold_summary_rows)
    if fold_summary_long.empty:
        raise ValueError("fold_summary_long is empty.")

    metric_cols = [
        "total_return",
        "annualized_return",
        "annualized_volatility",
        "sharpe_ratio",
        "max_drawdown",
        "average_turnover",
        "annualized_turnover",
    ]

    grouped = fold_summary_long.groupby("strategy")
    aggregate_mean = grouped[metric_cols].mean().add_prefix("mean_")
    aggregate_std = grouped[metric_cols].std().add_prefix("std_")
    aggregate_median = grouped[metric_cols].median().add_prefix("median_")

    benchmark_df = (
        fold_summary_long[fold_summary_long["strategy"] == benchmark_strategy_name][
            ["fold_id", "annualized_return", "sharpe_ratio"]
        ]
        .rename(
            columns={
                "annualized_return": "benchmark_ann_return",
                "sharpe_ratio": "benchmark_sharpe",
            }
        )
    )

    merged = fold_summary_long.merge(benchmark_df, on="fold_id", how="left")
    merged["win_vs_benchmark_return"] = (
        merged["annualized_return"] > merged["benchmark_ann_return"]
    ).astype(float)
    merged["win_vs_benchmark_sharpe"] = (
        merged["sharpe_ratio"] > merged["benchmark_sharpe"]
    ).astype(float)

    win_rate = merged.groupby("strategy")[
        ["win_vs_benchmark_return", "win_vs_benchmark_sharpe"]
    ].mean()

    aggregate_summary = pd.concat(
        [aggregate_mean, aggregate_std, aggregate_median, win_rate],
        axis=1,
    ).sort_values("mean_sharpe_ratio", ascending=False)

    return StaticBLRunResult(
        fold_results=fold_results,
        step_results=step_results,
        fold_summary_long=fold_summary_long,
        aggregate_summary=aggregate_summary,
    )
