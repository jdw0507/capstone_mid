from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
from sklearn.linear_model import SGDRegressor

from src.forecasting.base import BaseForecaster


class SGDRegressionForecaster(BaseForecaster):
    """
    SGDRegressor-based forecaster (paper-aligned ML model).
    """

    def __init__(
        self,
        loss: str = "huber",
        alpha: float = 1e-4,
        max_iter: int = 3000,
        tol: float = 1e-3,
        random_state: int = 42,
        model_name: Optional[str] = None,
        feature_columns: Optional[list[str]] = None,
        target_column: str = "target_5d_excess",
        horizon_days: int = 5,
        periods_per_year: int = 252,
    ) -> None:
        super().__init__(
            model_name=model_name or "SGDRegressor",
            feature_columns=feature_columns,
            target_column=target_column,
            horizon_days=horizon_days,
            periods_per_year=periods_per_year,
        )

        self.loss = loss
        self.alpha = alpha
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state

        self.model = SGDRegressor(
            loss=self.loss,
            alpha=self.alpha,
            max_iter=self.max_iter,
            tol=self.tol,
            random_state=self.random_state,
        )

        self.fill_values_: pd.Series | None = None
        self.global_residual_std_: float | None = None
        self.asset_residual_std_: pd.Series | None = None
        self.coef_: pd.Series | None = None
        self.intercept_: float | None = None

    def _fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        meta: Optional[pd.DataFrame] = None,
    ) -> None:
        X_fit = X.copy()
        y_fit = y.copy()

        self.fill_values_ = X_fit.median(numeric_only=True).reindex(X_fit.columns).fillna(0.0)
        X_fit = X_fit.fillna(self.fill_values_)

        self.model.fit(X_fit, y_fit)

        pred_in = pd.Series(self.model.predict(X_fit), index=y_fit.index)
        resid = y_fit - pred_in
        self.global_residual_std_ = float(resid.std(ddof=1)) if len(resid) > 1 else 0.0

        if meta is not None and "asset" in meta.columns:
            tmp = pd.DataFrame(
                {"asset": meta["asset"].reset_index(drop=True), "resid": resid.reset_index(drop=True)}
            )
            self.asset_residual_std_ = tmp.groupby("asset")["resid"].std(ddof=1).astype(float)
        else:
            self.asset_residual_std_ = None

        self.coef_ = pd.Series(self.model.coef_, index=X.columns, name="coef")
        self.intercept_ = float(self.model.intercept_[0]) if np.ndim(self.model.intercept_) > 0 else float(self.model.intercept_)

    def _predict(
        self,
        X: pd.DataFrame,
        meta: Optional[pd.DataFrame] = None,
    ) -> pd.Series | np.ndarray:
        if self.fill_values_ is None:
            raise ValueError("fill_values_ is missing. fit() first.")
        X_pred = X.copy().fillna(self.fill_values_)
        return self.model.predict(X_pred)

    def _predict_uncertainty(
        self,
        X: pd.DataFrame,
        meta: Optional[pd.DataFrame] = None,
    ) -> pd.Series | np.ndarray | None:
        if self.global_residual_std_ is None:
            return None

        if meta is not None and "asset" in meta.columns and self.asset_residual_std_ is not None:
            out = meta["asset"].map(self.asset_residual_std_).astype(float)
            out = out.fillna(self.global_residual_std_)
            return out.values

        return np.full(len(X), self.global_residual_std_, dtype=float)

