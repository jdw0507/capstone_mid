from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from .base import AllocationModel


class BlackLittermanAllocator(AllocationModel):
    """
    Allocate portfolio weights from BL posterior expected returns and covariance.

    Objective:
        maximize  mu'w - 0.5 * delta * w' Sigma w

    Equivalent minimization:
        minimize  0.5 * delta * w' Sigma w - mu'w
    """

    def __init__(
        self,
        risk_aversion: float = 2.5,
        long_only: bool = True,
        weight_bounds: Optional[Tuple[float, float]] = None,
        reference_weights: Optional[pd.Series] = None,
        reference_weight_penalty: float = 0.0,
        l2_reg: float = 0.0,
    ) -> None:
        self.risk_aversion = risk_aversion
        self.long_only = long_only
        self.weight_bounds = weight_bounds
        self.reference_weights = reference_weights
        self.reference_weight_penalty = reference_weight_penalty
        self.l2_reg = l2_reg

        self.last_objective_value_: float | None = None
        self.last_reference_weights_: pd.Series | None = None

    def optimize(
        self,
        expected_returns: pd.Series,
        covariance: pd.DataFrame,
    ) -> pd.Series:
        self._validate_inputs(expected_returns, covariance)

        assets = expected_returns.index.tolist()
        mu = expected_returns.loc[assets].astype(float).values
        cov = covariance.loc[assets, assets].astype(float).values
        n_assets = len(assets)

        ref_w = None
        if self.reference_weights is not None:
            ref_w = self.reference_weights.reindex(assets).astype(float).fillna(0.0)
            if not np.isclose(ref_w.sum(), 0.0):
                ref_w = ref_w / ref_w.sum()
            else:
                ref_w = None

        self.last_reference_weights_ = None if ref_w is None else ref_w.copy()

        # Unconstrained closed-form solution
        if (
            not self.long_only
            and self.weight_bounds is None
            and self.reference_weight_penalty == 0.0
            and self.l2_reg == 0.0
        ):
            try:
                w = np.linalg.solve(self.risk_aversion * cov, mu)
                w = pd.Series(w, index=assets, name="weight")
                if not np.isclose(w.sum(), 0.0):
                    w = w / w.sum()
                self.last_objective_value_ = self._objective(
                    w.values,
                    mu=mu,
                    cov=cov,
                    ref_w=None if ref_w is None else ref_w.values,
                )
                return w
            except np.linalg.LinAlgError:
                pass

        x0 = np.ones(n_assets) / n_assets
        bounds = self._build_bounds(n_assets)
        constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]

        result = minimize(
            fun=lambda w: self._objective(
                w=w,
                mu=mu,
                cov=cov,
                ref_w=None if ref_w is None else ref_w.values,
            ),
            x0=x0,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
        )

        if not result.success:
            raise ValueError(f"BlackLittermanAllocator optimization failed: {result.message}")

        w = pd.Series(result.x, index=assets, name="weight")
        w = w / w.sum()
        self.last_objective_value_ = float(result.fun)
        return w

    def _objective(
        self,
        w: np.ndarray,
        mu: np.ndarray,
        cov: np.ndarray,
        ref_w: np.ndarray | None = None,
    ) -> float:
        risk_term = 0.5 * self.risk_aversion * float(w.T @ cov @ w)
        return_term = float(mu @ w)
        l2_term = 0.5 * self.l2_reg * float(w @ w)

        ref_term = 0.0
        if ref_w is not None and self.reference_weight_penalty > 0:
            diff = w - ref_w
            ref_term = 0.5 * self.reference_weight_penalty * float(diff @ diff)

        return risk_term - return_term + l2_term + ref_term

    def _build_bounds(self, n_assets: int):
        if self.weight_bounds is not None:
            return [self.weight_bounds] * n_assets

        if self.long_only:
            return [(0.0, 1.0)] * n_assets

        return [(None, None)] * n_assets

    def _validate_inputs(
        self,
        expected_returns: pd.Series,
        covariance: pd.DataFrame,
    ) -> None:
        if not isinstance(expected_returns, pd.Series):
            raise TypeError("expected_returns must be a pandas Series.")
        if not isinstance(covariance, pd.DataFrame):
            raise TypeError("covariance must be a pandas DataFrame.")
        if expected_returns.empty:
            raise ValueError("expected_returns is empty.")
        if covariance.empty:
            raise ValueError("covariance is empty.")
        if not covariance.index.equals(covariance.columns):
            raise ValueError("covariance index and columns must match.")

        assets = expected_returns.index.tolist()
        missing = [a for a in assets if a not in covariance.index]
        if missing:
            raise ValueError(f"Assets missing in covariance: {missing}")