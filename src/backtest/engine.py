from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.data.preprocessing import (
    prepare_price_data_for_allocation,
    compute_simple_returns,
)
from .rebalancing import build_rebalance_schedule
from .performance import make_performance_table


@dataclass
class BacktestResult:
    portfolio_returns: pd.Series
    portfolio_gross_returns: pd.Series
    portfolio_turnover: pd.Series

    benchmark_returns: pd.Series
    benchmark_gross_returns: pd.Series
    benchmark_turnover: pd.Series

    weights_history: pd.DataFrame
    benchmark_weights_history: pd.DataFrame

    monthly_prices: pd.DataFrame
    monthly_returns: pd.DataFrame
    schedule: pd.DataFrame
    summary: pd.DataFrame


class BacktestEngine:
    """
    월별 리밸런싱 기반 백테스트 엔진 (v1)

    흐름:
    1. 가격 데이터 -> 월말 가격
    2. 리밸런싱 날짜 생성
    3. 각 리밸런싱 시점마다 과거 lookback 구간만 사용해서
       기대수익률 / 공분산 추정
    4. 최적 비중 계산
    5. 다음 한 기간의 실제 수익률 반영
    """

    def __init__(
        self,
        return_model,
        risk_model,
        allocation_model,
        lookback_periods: int = 36,
        transaction_cost: float = 0.001,
        resample_freq: str = "ME",
        periods_per_year: int = 12,
        risk_free_rate: float = 0.0,
        charge_initial_cost: bool = False,
        debug: bool = False,
        debug_max_steps: int | None = None,
    ) -> None:
        self.return_model = return_model
        self.risk_model = risk_model
        self.allocation_model = allocation_model

        self.lookback_periods = lookback_periods
        self.transaction_cost = transaction_cost
        self.resample_freq = resample_freq
        self.periods_per_year = periods_per_year
        self.risk_free_rate = risk_free_rate
        self.charge_initial_cost = charge_initial_cost

        self.debug = debug
        self.debug_max_steps = debug_max_steps

    def run(self, prices: pd.DataFrame) -> BacktestResult:
        monthly_prices = prepare_price_data_for_allocation(
            prices=prices,
            use_common_period=True,
            resample_freq=self.resample_freq,
        )

        monthly_returns = compute_simple_returns(monthly_prices)

        schedule = build_rebalance_schedule(
            monthly_prices=monthly_prices,
            lookback_periods=self.lookback_periods,
        )

        portfolio_net_return_map: dict[pd.Timestamp, float] = {}
        portfolio_gross_return_map: dict[pd.Timestamp, float] = {}
        portfolio_turnover_map: dict[pd.Timestamp, float] = {}

        benchmark_net_return_map: dict[pd.Timestamp, float] = {}
        benchmark_gross_return_map: dict[pd.Timestamp, float] = {}
        benchmark_turnover_map: dict[pd.Timestamp, float] = {}

        weights_records: list[pd.Series] = []
        benchmark_weight_records: list[pd.Series] = []

        current_strategy_weights: pd.Series | None = None
        current_benchmark_weights: pd.Series | None = None

        debug_counter = 0

        for row in schedule.itertuples(index=False):
            rebalance_date = row.rebalance_date
            next_date = row.next_date

            # 훈련 데이터는 rebalance_date까지만 사용 -> 룩어헤드 방지
            train_prices = monthly_prices.loc[:rebalance_date].tail(self.lookback_periods + 1)
            train_returns = compute_simple_returns(train_prices)

            mu = self.return_model.fit_predict(train_returns).sort_index()
            cov = self.risk_model.fit_predict(train_returns)
            cov = cov.loc[mu.index, mu.index]

            target_weights = self.allocation_model.optimize(mu, cov).sort_index()
            target_weights.name = rebalance_date
            weights_records.append(target_weights)

            realized_asset_returns = monthly_returns.loc[next_date, target_weights.index].astype(float)

            strategy_turnover = self._compute_turnover(
                current_weights=current_strategy_weights,
                target_weights=target_weights,
                charge_initial_cost=self.charge_initial_cost,
            )
            strategy_gross_return = float(target_weights @ realized_asset_returns)
            strategy_net_return = strategy_gross_return - strategy_turnover * self.transaction_cost

            portfolio_turnover_map[next_date] = strategy_turnover
            portfolio_gross_return_map[next_date] = strategy_gross_return
            portfolio_net_return_map[next_date] = strategy_net_return

            # debug 출력
            if self.debug and (
                self.debug_max_steps is None or debug_counter < self.debug_max_steps
            ):
                self._log_step(
                    model_name="Strategy",
                    rebalance_date=rebalance_date,
                    next_date=next_date,
                    train_returns=train_returns,
                    mu=mu,
                    target_weights=target_weights,
                    realized_asset_returns=realized_asset_returns,
                    turnover=strategy_turnover,
                    gross_return=strategy_gross_return,
                    net_return=strategy_net_return,
                )
                debug_counter += 1

            current_strategy_weights = self._drift_weights(
                target_weights=target_weights,
                realized_returns=realized_asset_returns,
            )

            # Equal Weight benchmark
            benchmark_target_weights = self._equal_weight(target_weights.index)
            benchmark_target_weights.name = rebalance_date
            benchmark_weight_records.append(benchmark_target_weights)

            benchmark_turnover = self._compute_turnover(
                current_weights=current_benchmark_weights,
                target_weights=benchmark_target_weights,
                charge_initial_cost=self.charge_initial_cost,
            )
            benchmark_gross_return = float(benchmark_target_weights @ realized_asset_returns)
            benchmark_net_return = benchmark_gross_return - benchmark_turnover * self.transaction_cost

            benchmark_turnover_map[next_date] = benchmark_turnover
            benchmark_gross_return_map[next_date] = benchmark_gross_return
            benchmark_net_return_map[next_date] = benchmark_net_return

            current_benchmark_weights = self._drift_weights(
                target_weights=benchmark_target_weights,
                realized_returns=realized_asset_returns,
            )

        portfolio_returns = pd.Series(portfolio_net_return_map, name="portfolio_return").sort_index()
        portfolio_gross_returns = pd.Series(
            portfolio_gross_return_map, name="portfolio_gross_return"
        ).sort_index()
        portfolio_turnover = pd.Series(portfolio_turnover_map, name="portfolio_turnover").sort_index()

        benchmark_returns = pd.Series(
            benchmark_net_return_map, name="equal_weight_return"
        ).sort_index()
        benchmark_gross_returns = pd.Series(
            benchmark_gross_return_map, name="equal_weight_gross_return"
        ).sort_index()
        benchmark_turnover = pd.Series(
            benchmark_turnover_map, name="equal_weight_turnover"
        ).sort_index()

        weights_history = pd.DataFrame(weights_records)
        weights_history.index.name = "rebalance_date"
        weights_history = weights_history.sort_index().sort_index(axis=1)

        benchmark_weights_history = pd.DataFrame(benchmark_weight_records)
        benchmark_weights_history.index.name = "rebalance_date"
        benchmark_weights_history = benchmark_weights_history.sort_index().sort_index(axis=1)

        summary = make_performance_table(
            returns_map={
                "mvo_strategy": portfolio_returns,
                "equal_weight": benchmark_returns,
            },
            turnover_map={
                "mvo_strategy": portfolio_turnover,
                "equal_weight": benchmark_turnover,
            },
            periods_per_year=self.periods_per_year,
            risk_free_rate=self.risk_free_rate,
        )
        return BacktestResult(
            portfolio_returns=portfolio_returns,
            portfolio_gross_returns=portfolio_gross_returns,
            portfolio_turnover=portfolio_turnover,
            benchmark_returns=benchmark_returns,
            benchmark_gross_returns=benchmark_gross_returns,
            benchmark_turnover=benchmark_turnover,
            weights_history=weights_history,
            benchmark_weights_history=benchmark_weights_history,
            monthly_prices=monthly_prices,
            monthly_returns=monthly_returns,
            schedule=schedule,
            summary=summary,
            fitted_return_model=self.return_model,
            fitted_risk_model=self.risk_model,
            fitted_allocation_model=self.allocation_model,
        )

    def _equal_weight(self, assets) -> pd.Series:
        assets = list(assets)
        n_assets = len(assets)
        return pd.Series(1.0 / n_assets, index=assets, name="weight").sort_index()

    def _compute_turnover(
        self,
        current_weights: pd.Series | None,
        target_weights: pd.Series,
        charge_initial_cost: bool = False,
    ) -> float:
        """
        one-way turnover 기준:
        - 초기 투자: charge_initial_cost=True면 100% 매수로 간주 -> turnover = 1.0
        - 이후 리밸런싱: 0.5 * sum(|w_target - w_current|)
        """
        target = target_weights.sort_index()

        if current_weights is None:
            return 1.0 if charge_initial_cost else 0.0

        current = current_weights.reindex(target.index).fillna(0.0).sort_index()
        turnover = 0.5 * float(np.abs(target - current).sum())
        return turnover

    def _drift_weights(
        self,
        target_weights: pd.Series,
        realized_returns: pd.Series,
    ) -> pd.Series:
        """
        한 기간 수익률 반영 후 다음 리밸런싱 직전의 자연 드리프트된 비중
        """
        target = target_weights.sort_index()
        realized = realized_returns.reindex(target.index).astype(float).sort_index()

        end_values = target * (1.0 + realized)
        total_value = float(end_values.sum())

        if total_value <= 0:
            raise ValueError("포트폴리오 가치가 0 이하가 되어 weight drift를 계산할 수 없습니다.")

        drifted_weights = end_values / total_value
        drifted_weights.name = "weight"
        return drifted_weights

    def _log_step(
        self,
        model_name: str,
        rebalance_date: pd.Timestamp,
        next_date: pd.Timestamp,
        train_returns: pd.DataFrame,
        mu: pd.Series,
        target_weights: pd.Series,
        realized_asset_returns: pd.Series,
        turnover: float,
        gross_return: float,
        net_return: float,
    ) -> None:
        print("\n" + "=" * 90)
        print(f"[{model_name}] REBALANCE")
        print(f"rebalance_date : {pd.Timestamp(rebalance_date).date()}")
        print(f"next_date      : {pd.Timestamp(next_date).date()}")
        print(f"train window   : {train_returns.index.min().date()} ~ {train_returns.index.max().date()}")
        print(f"n_train_obs    : {len(train_returns)}")

        print("\n--- Expected Returns (mu) ---")
        print(mu.round(6).to_string())

        print("\n--- Target Weights ---")
        print(target_weights.round(6).to_string())

        print("\n--- Next Period Asset Returns ---")
        print(realized_asset_returns.round(6).to_string())

        print("\n--- Portfolio Step Stats ---")
        print(f"turnover       : {turnover:.6f}")
        print(f"gross_return   : {gross_return:.6f}")
        print(f"net_return     : {net_return:.6f}")
        print("=" * 90)

@dataclass
class BacktestResult:
    portfolio_returns: pd.Series
    portfolio_gross_returns: pd.Series
    portfolio_turnover: pd.Series

    benchmark_returns: pd.Series
    benchmark_gross_returns: pd.Series
    benchmark_turnover: pd.Series

    weights_history: pd.DataFrame
    benchmark_weights_history: pd.DataFrame

    monthly_prices: pd.DataFrame
    monthly_returns: pd.DataFrame
    schedule: pd.DataFrame
    summary: pd.DataFrame

    fitted_return_model: object | None = None
    fitted_risk_model: object | None = None
    fitted_allocation_model: object | None = None