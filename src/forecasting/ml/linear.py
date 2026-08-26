from __future__ import annotations

from typing import Literal, Optional

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge, Lasso

from src.forecasting.base import BaseForecaster


LinearModelType = Literal["ols", "ridge", "lasso"]


class LinearForecaster(BaseForecaster):
    """
    선형회귀 baseline forecaster

    Parameters
    ----------
    linear_type : {"ols", "ridge", "lasso"}
    alpha : float
        ridge/lasso regularization strength
    positive : bool
        계수 양수 제약 여부
    """

    def __init__(
        self,
        linear_type: LinearModelType = "ols",
        alpha: float = 1.0,
        positive: bool = False,
        model_name: Optional[str] = None,
        feature_columns: Optional[list[str]] = None,
        target_column: str = "target_5d_excess",
        horizon_days: int = 5,
        periods_per_year: int = 252,
    ) -> None:
        super().__init__(
            model_name=model_name or f"Linear_{linear_type}",
            feature_columns=feature_columns,
            target_column=target_column,
            horizon_days=horizon_days,
            periods_per_year=periods_per_year,
        )

        self.linear_type = linear_type
        self.alpha = alpha
        self.positive = positive

        self.model = self._build_model()
        self.global_residual_std_: float | None = None
        self.asset_residual_std_: pd.Series | None = None
        self.coef_: pd.Series | None = None
        self.intercept_: float | None = None

    def _build_model(self):
        if self.linear_type == "ols":
            return LinearRegression(positive=self.positive)
        elif self.linear_type == "ridge":
            return Ridge(alpha=self.alpha, positive=self.positive)
        elif self.linear_type == "lasso":
            return Lasso(alpha=self.alpha, positive=self.positive, max_iter=10000)
        else:
            raise ValueError(f"지원하지 않는 linear_type: {self.linear_type}")

    def _fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        meta: Optional[pd.DataFrame] = None,
    ) -> None:
        X_fit = X.copy()
        y_fit = y.copy()

        # 단순 baseline이므로 missing feature는 0으로 대체
        X_fit = X_fit.fillna(0.0)

        self.model.fit(X_fit, y_fit)

        pred_in = pd.Series(self.model.predict(X_fit), index=y_fit.index)
        resid = y_fit - pred_in

        self.global_residual_std_ = float(resid.std(ddof=1)) if len(resid) > 1 else 0.0

        if meta is not None and "asset" in meta.columns:
            tmp = pd.DataFrame({
                "asset": meta["asset"].reset_index(drop=True),
                "resid": resid.reset_index(drop=True),
            })
            asset_std = tmp.groupby("asset")["resid"].std(ddof=1)
            self.asset_residual_std_ = asset_std.astype(float)
        else:
            self.asset_residual_std_ = None

        self.coef_ = pd.Series(self.model.coef_, index=X.columns, name="coef")
        self.intercept_ = float(self.model.intercept_)

    def _predict(
        self,
        X: pd.DataFrame,
        meta: Optional[pd.DataFrame] = None,
    ) -> pd.Series | np.ndarray:
        X_pred = X.copy().fillna(0.0)
        pred = self.model.predict(X_pred)
        return pred

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

        return np.full(shape=len(X), fill_value=self.global_residual_std_, dtype=float)