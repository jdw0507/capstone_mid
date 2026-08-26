from __future__ import annotations

from abc import ABC
from typing import Optional

import pandas as pd
import statsmodels.api as sm

from .base import ExpectedReturnModel


class LinearFactorExpectedReturnModel(ExpectedReturnModel, ABC):
    """
    선형 팩터 회귀 기반 기대수익률 모델 공통 베이스

    expected_return_monthly
      = E[RF] + alpha(optional) + beta' * E[factors]

    최종 출력은 periods_per_year를 곱한 연율 기대수익률
    """

    factor_columns: tuple[str, ...] = ()
    risk_free_column: str = "RF"

    def __init__(
        self,
        factors: pd.DataFrame,
        periods_per_year: int = 12,
        min_obs: int = 24,
        include_alpha: bool = False,
        premium_window: Optional[int] = None,
    ) -> None:
        self.factors = self._prepare_factors(factors)
        self.periods_per_year = periods_per_year
        self.min_obs = min_obs
        self.include_alpha = include_alpha
        self.premium_window = premium_window

        self.last_factor_premiums_: pd.Series | None = None
        self.last_rf_: float | None = None
        self.last_regression_diagnostics_: pd.DataFrame | None = None

    def fit_predict(self, returns: pd.DataFrame) -> pd.Series:
        asset_returns = self._prepare_returns(returns)
        aligned_factors = self._align_factors(asset_returns.index)

        premium_sample = aligned_factors.copy()
        if self.premium_window is not None:
            premium_sample = premium_sample.tail(self.premium_window)

        expected_factor_premiums = premium_sample[list(self.factor_columns)].mean()
        expected_rf = float(premium_sample[self.risk_free_column].mean())

        expected_returns = {}
        diagnostics = []

        for asset in asset_returns.columns:
            reg_df = pd.concat(
                [
                    asset_returns[[asset]],
                    aligned_factors[list(self.factor_columns) + [self.risk_free_column]],
                ],
                axis=1,
            ).dropna()

            if len(reg_df) < self.min_obs:
                continue

            y = reg_df[asset] - reg_df[self.risk_free_column]
            X = sm.add_constant(reg_df[list(self.factor_columns)], has_constant="add")

            result = sm.OLS(y, X, missing="drop").fit()

            alpha = float(result.params.get("const", 0.0))
            betas = result.params.reindex(self.factor_columns).fillna(0.0)

            expected_excess_monthly = float(betas @ expected_factor_premiums)
            if self.include_alpha:
                expected_excess_monthly += alpha

            expected_monthly = expected_rf + expected_excess_monthly
            expected_annualized = expected_monthly * self.periods_per_year

            expected_returns[asset] = expected_annualized

            row = {
                "asset": asset,
                "alpha_monthly": alpha,
                "expected_monthly": expected_monthly,
                "expected_annualized": expected_annualized,
                "r_squared": float(result.rsquared),
                "n_obs": int(result.nobs),
            }

            for factor_name in self.factor_columns:
                row[f"beta_{factor_name.lower()}"] = float(betas[factor_name])

            diagnostics.append(row)

        if not expected_returns:
            raise ValueError("min_obs 조건을 만족하는 자산이 없습니다.")

        self.last_factor_premiums_ = expected_factor_premiums
        self.last_rf_ = expected_rf
        self.last_regression_diagnostics_ = (
            pd.DataFrame(diagnostics).set_index("asset").sort_index()
        )

        mu = pd.Series(expected_returns, name="expected_return").sort_index()
        return mu

    def _prepare_factors(self, factors: pd.DataFrame) -> pd.DataFrame:
        if not isinstance(factors, pd.DataFrame):
            raise TypeError("factors는 pandas DataFrame이어야 합니다.")

        if factors.empty:
            raise ValueError("factors가 비어 있습니다.")

        required = list(self.factor_columns) + [self.risk_free_column]
        missing_cols = [c for c in required if c not in factors.columns]
        if missing_cols:
            raise ValueError(f"factors에 필요한 컬럼이 없습니다: {missing_cols}")

        out = factors.copy()
        out.index = pd.to_datetime(out.index)
        out = out.sort_index()
        out = out[~out.index.duplicated(keep="first")]
        out = out.apply(pd.to_numeric, errors="coerce")
        out = out[required].dropna(how="all")

        return out

    def _prepare_returns(self, returns: pd.DataFrame) -> pd.DataFrame:
        if not isinstance(returns, pd.DataFrame):
            raise TypeError("returns는 pandas DataFrame이어야 합니다.")

        if returns.empty:
            raise ValueError("returns가 비어 있습니다.")

        out = returns.copy()
        out.index = pd.to_datetime(out.index)
        out = out.sort_index()
        out = out[~out.index.duplicated(keep="first")]
        out = out.apply(pd.to_numeric, errors="coerce")

        return out

    def _align_factors(self, return_index: pd.Index) -> pd.DataFrame:
        common_index = pd.Index(return_index).intersection(self.factors.index)
        if len(common_index) == 0:
            raise ValueError("자산수익률과 팩터 데이터의 공통 날짜가 없습니다.")

        aligned = self.factors.loc[common_index].sort_index()

        if len(aligned) < self.min_obs:
            raise ValueError(
                f"팩터 데이터의 공통 관측치가 부족합니다. "
                f"현재 {len(aligned)}개, 최소 {self.min_obs}개 필요"
            )

        return aligned