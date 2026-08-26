from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from .base import AllocationModel


class RiskParity(AllocationModel):
    """
    Long-only Risk Parity allocator

    Parameters
    ----------
    weight_bounds : tuple[float, float] | None
        각 자산 비중 하한/상한
        예: (0.0, 1.0), (0.0, 0.3)
    target_risk_budget : pd.Series | None
        자산별 목표 리스크 예산.
        None이면 동일 리스크 예산(1/N)
    """

    def __init__(
        self,
        weight_bounds: Optional[Tuple[float, float]] = None,
        target_risk_budget: Optional[pd.Series] = None,
    ) -> None:
        self.weight_bounds = weight_bounds
        self.target_risk_budget = target_risk_budget

    def optimize(
        self,
        expected_returns: pd.Series,
        covariance: pd.DataFrame,
    ) -> pd.Series:
        """
        base interface를 맞추기 위해 expected_returns를 받지만,
        실제 Risk Parity 계산에는 covariance만 사용한다.
        """
        self._validate_inputs(covariance)

        assets = covariance.index.tolist()
        cov = covariance.loc[assets, assets].values
        n_assets = len(assets)

        x0 = np.ones(n_assets) / n_assets
        bounds = self._build_bounds(n_assets)

        if self.target_risk_budget is None:
            b = np.ones(n_assets) / n_assets
        else:
            b = (
                self.target_risk_budget.reindex(assets)
                .astype(float)
                .fillna(0.0)
                .values
            )
            if np.isclose(b.sum(), 0.0):
                raise ValueError("target_risk_budget 합이 0입니다.")
            b = b / b.sum()

        constraints = [
            {"type": "eq", "fun": lambda w: np.sum(w) - 1.0},
        ]

        result = minimize(
            fun=lambda w: self._risk_parity_objective(w, cov, b),
            x0=x0,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
        )

        if not result.success:
            raise ValueError(f"Risk Parity 최적화 실패: {result.message}")

        weights = pd.Series(result.x, index=assets, name="weight")
        weights = weights / weights.sum()

        return weights

    def _portfolio_volatility(self, w: np.ndarray, cov: np.ndarray) -> float:
        var = float(w.T @ cov @ w)
        var = max(var, 1e-16)
        return float(np.sqrt(var))

    def _risk_contributions(self, w: np.ndarray, cov: np.ndarray) -> np.ndarray:
        """
        각 자산의 total risk contribution
        """
        sigma = self._portfolio_volatility(w, cov)
        mrc = (cov @ w) / sigma
        rc = w * mrc
        return rc

    def _risk_parity_objective(
        self,
        w: np.ndarray,
        cov: np.ndarray,
        target_budget: np.ndarray,
    ) -> float:
        """
        실제 RC 비율과 목표 리스크 예산 비율의 차이를 최소화
        """
        sigma = self._portfolio_volatility(w, cov)
        rc = self._risk_contributions(w, cov)

        # 비율 기준 비교
        rc_ratio = rc / sigma

        return float(np.sum((rc_ratio - target_budget) ** 2))

    def _build_bounds(self, n_assets: int):
        if self.weight_bounds is not None:
            return [self.weight_bounds] * n_assets
        return [(0.0, 1.0)] * n_assets

    def _validate_inputs(self, covariance: pd.DataFrame) -> None:
        if not isinstance(covariance, pd.DataFrame):
            raise TypeError("covariance는 pandas DataFrame이어야 합니다.")

        if covariance.empty:
            raise ValueError("covariance가 비어 있습니다.")

        if covariance.isna().any().any():
            raise ValueError("covariance에 NaN이 있습니다.")

        if not covariance.index.equals(covariance.columns):
            raise ValueError("covariance의 index와 columns가 일치해야 합니다.")
        
def compute_risk_contributions(
    weights: pd.Series,
    covariance: pd.DataFrame,
) -> pd.Series:
    assets = weights.index.tolist()
    w = weights.loc[assets].values
    cov = covariance.loc[assets, assets].values

    var = float(w.T @ cov @ w)
    var = max(var, 1e-16)
    sigma = np.sqrt(var)

    mrc = (cov @ w) / sigma
    rc = w * mrc

    return pd.Series(rc, index=assets, name="risk_contribution")