from __future__ import annotations

from typing import Literal, Optional, Tuple

import numpy as np
import pandas as pd
import warnings
from scipy.optimize import minimize

from .base import AllocationModel


ObjectiveType = Literal["max_sharpe", "min_variance"]


class MVO(AllocationModel):
    """
    Mean-Variance Optimization (MVO)

    Parameters
    ----------
    objective : {"max_sharpe", "min_variance"}
        최적화 목적함수
    risk_free_rate : float
        max_sharpe에서 사용할 무위험수익률
    long_only : bool
        True면 공매도 금지
    weight_bounds : tuple[float, float] | None
        각 자산 비중 하한/상한
        예: (0.0, 1.0), (0.0, 0.5)
    """

    def __init__(
        self,
        objective: ObjectiveType = "max_sharpe",
        risk_free_rate: float = 0.0,
        long_only: bool = True,
        weight_bounds: Optional[Tuple[float, float]] = None,
    ) -> None:
        self.objective = objective
        self.risk_free_rate = risk_free_rate
        self.long_only = long_only
        self.weight_bounds = weight_bounds

    def optimize(
        self,
        expected_returns: pd.Series,
        covariance: pd.DataFrame,
    ) -> pd.Series:
        self._validate_inputs(expected_returns, covariance)

        assets = expected_returns.index.tolist()
        mu = expected_returns.loc[assets].values
        cov = covariance.loc[assets, assets].values
        n_assets = len(assets)

        x0 = np.ones(n_assets) / n_assets

        constraints = [
            {"type": "eq", "fun": lambda w: np.sum(w) - 1.0}
        ]

        bounds = self._build_bounds(n_assets)

        # 1) primary solve
        result = self._solve_slsqp(
            objective=self.objective,
            x0=x0,
            mu=mu,
            cov=cov,
            bounds=bounds,
            constraints=constraints,
        )
        if result.success:
            return self._normalize_weights(result.x, assets)

        # 2) retry with ridge-regularized covariance
        cov_retry = cov + self._ridge_eps(cov) * np.eye(n_assets)
        result_retry = self._solve_slsqp(
            objective=self.objective,
            x0=x0,
            mu=mu,
            cov=cov_retry,
            bounds=bounds,
            constraints=constraints,
        )
        if result_retry.success:
            warnings.warn(
                f"MVO primary optimization failed ({result.message}); recovered with covariance regularization.",
                RuntimeWarning,
            )
            return self._normalize_weights(result_retry.x, assets)

        # 3) if max_sharpe, fallback once to min_variance
        if self.objective == "max_sharpe":
            result_mv = self._solve_slsqp(
                objective="min_variance",
                x0=x0,
                mu=mu,
                cov=cov_retry,
                bounds=bounds,
                constraints=constraints,
            )
            if result_mv.success:
                warnings.warn(
                    (
                        "MVO max_sharpe optimization failed "
                        f"({result_retry.message}); fell back to min_variance."
                    ),
                    RuntimeWarning,
                )
                return self._normalize_weights(result_mv.x, assets)

        # 4) last-resort feasible equal-like weights
        w_fallback = self._build_feasible_fallback_weights(n_assets, bounds)
        warnings.warn(
            (
                "MVO optimization failed; using feasible fallback weights. "
                f"primary='{result.message}', retry='{result_retry.message}'"
            ),
            RuntimeWarning,
        )
        return self._normalize_weights(w_fallback, assets)

    def _solve_slsqp(
        self,
        objective: ObjectiveType,
        x0: np.ndarray,
        mu: np.ndarray,
        cov: np.ndarray,
        bounds,
        constraints,
    ):
        if objective == "max_sharpe":
            objective_fn = lambda w: self._negative_sharpe(w, mu, cov)
        elif objective == "min_variance":
            objective_fn = lambda w: self._portfolio_variance(w, cov)
        else:
            raise ValueError(f"지원하지 않는 objective: {objective}")

        return minimize(
            fun=objective_fn,
            x0=x0,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
        )

    def _ridge_eps(self, cov: np.ndarray) -> float:
        diag = np.diag(cov)
        finite_diag = diag[np.isfinite(diag)]
        scale = float(np.nanmedian(np.abs(finite_diag))) if finite_diag.size > 0 else 1.0
        scale = max(scale, 1.0)
        return 1e-6 * scale

    def _normalize_weights(self, w: np.ndarray, assets: list[str]) -> pd.Series:
        out = pd.Series(w, index=assets, name="weight").astype(float)
        s = float(out.sum())
        if np.isclose(s, 0.0):
            return pd.Series(1.0 / len(assets), index=assets, name="weight")
        out = out / s
        return out

    def _build_feasible_fallback_weights(self, n_assets: int, bounds) -> np.ndarray:
        if bounds is None:
            return np.ones(n_assets, dtype=float) / n_assets

        lower = np.array([b[0] if b[0] is not None else -np.inf for b in bounds], dtype=float)
        upper = np.array([b[1] if b[1] is not None else np.inf for b in bounds], dtype=float)

        lb_sum = float(np.sum(lower[np.isfinite(lower)]))
        ub_sum = float(np.sum(upper[np.isfinite(upper)]))
        if lb_sum > 1.0 + 1e-12 or ub_sum < 1.0 - 1e-12:
            raise ValueError("MVO bounds are infeasible for sum-to-one constraint.")

        w = np.ones(n_assets, dtype=float) / n_assets
        w = np.maximum(w, np.where(np.isfinite(lower), lower, w))
        w = np.minimum(w, np.where(np.isfinite(upper), upper, w))

        deficit = 1.0 - float(w.sum())
        if abs(deficit) <= 1e-12:
            return w

        if deficit > 0:
            slack = np.where(np.isfinite(upper), upper - w, 0.0)
            total_slack = float(np.sum(np.maximum(slack, 0.0)))
            if total_slack <= 1e-12:
                raise ValueError("Cannot distribute positive deficit under bounds.")
            w = w + deficit * np.maximum(slack, 0.0) / total_slack
        else:
            slack = np.where(np.isfinite(lower), w - lower, 0.0)
            total_slack = float(np.sum(np.maximum(slack, 0.0)))
            if total_slack <= 1e-12:
                raise ValueError("Cannot remove excess weight under bounds.")
            w = w + deficit * np.maximum(slack, 0.0) / total_slack

        return w

    def _build_bounds(self, n_assets: int):
        if self.weight_bounds is not None:
            lower, upper = self.weight_bounds
            return [self.weight_bounds] * n_assets

        if self.long_only:
            return [(0.0, 1.0)] * n_assets

        return [(-1.0, 1.0)] * n_assets

    def _portfolio_return(self, weights: np.ndarray, mu: np.ndarray) -> float:
        return float(weights @ mu)

    def _portfolio_variance(self, weights: np.ndarray, cov: np.ndarray) -> float:
        return float(weights.T @ cov @ weights)

    def _portfolio_volatility(self, weights: np.ndarray, cov: np.ndarray) -> float:
        variance = self._portfolio_variance(weights, cov)
        variance = max(variance, 1e-12)
        return float(np.sqrt(variance))

    def _negative_sharpe(
        self,
        weights: np.ndarray,
        mu: np.ndarray,
        cov: np.ndarray,
    ) -> float:
        port_return = self._portfolio_return(weights, mu)
        port_vol = self._portfolio_volatility(weights, cov)
        sharpe = (port_return - self.risk_free_rate) / port_vol
        return -float(sharpe)

    def _validate_inputs(
        self,
        expected_returns: pd.Series,
        covariance: pd.DataFrame,
    ) -> None:
        if not isinstance(expected_returns, pd.Series):
            raise TypeError("expected_returns는 pandas Series여야 합니다.")

        if not isinstance(covariance, pd.DataFrame):
            raise TypeError("covariance는 pandas DataFrame이어야 합니다.")

        if expected_returns.empty:
            raise ValueError("expected_returns가 비어 있습니다.")

        if covariance.empty:
            raise ValueError("covariance가 비어 있습니다.")

        if expected_returns.isna().any():
            raise ValueError("expected_returns에 NaN이 있습니다.")

        if covariance.isna().any().any():
            raise ValueError("covariance에 NaN이 있습니다.")

        if set(expected_returns.index) != set(covariance.index) or set(covariance.index) != set(covariance.columns):
            raise ValueError("expected_returns와 covariance의 자산 구성이 일치해야 합니다.")
