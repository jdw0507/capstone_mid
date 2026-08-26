from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from src.forecasting.base import BaseForecaster

try:
    from lightgbm import LGBMRegressor
except ImportError:  # pragma: no cover
    LGBMRegressor = None


class LightGBMForecaster(BaseForecaster):
    """
    LightGBM baseline forecaster

    특징
    ----
    - 빠르고 강한 tabular baseline
    - missing은 training median으로 보정
    - uncertainty는 training residual std proxy 사용
    """

    def __init__(
        self,
        n_estimators: int = 400,
        learning_rate: float = 0.03,
        num_leaves: int = 31,
        max_depth: int = -1,
        min_child_samples: int = 20,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        reg_alpha: float = 0.0,
        reg_lambda: float = 1.0,
        random_state: int = 42,
        model_name: Optional[str] = None,
        feature_columns: Optional[list[str]] = None,
        target_column: str = "target_5d_excess",
        horizon_days: int = 5,
        periods_per_year: int = 252,
    ) -> None:
        super().__init__(
            model_name=model_name or "LightGBM",
            feature_columns=feature_columns,
            target_column=target_column,
            horizon_days=horizon_days,
            periods_per_year=periods_per_year,
        )

        if LGBMRegressor is None:
            raise ImportError(
                "lightgbm이 설치되어 있지 않습니다. "
                "설치: C:/Anaconda/envs/torch-gpu/python.exe -m pip install lightgbm"
            )

        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.num_leaves = num_leaves
        self.max_depth = max_depth
        self.min_child_samples = min_child_samples
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.reg_alpha = reg_alpha
        self.reg_lambda = reg_lambda
        self.random_state = random_state

        self.model = LGBMRegressor(
            n_estimators=self.n_estimators,
            learning_rate=self.learning_rate,
            num_leaves=self.num_leaves,
            max_depth=self.max_depth,
            min_child_samples=self.min_child_samples,
            subsample=self.subsample,
            colsample_bytree=self.colsample_bytree,
            reg_alpha=self.reg_alpha,
            reg_lambda=self.reg_lambda,
            objective="regression",
            random_state=self.random_state,
            n_jobs=-1,
            verbosity=-1,
        )

        self.fill_values_: pd.Series | None = None
        self.global_residual_std_: float | None = None
        self.asset_residual_std_: pd.Series | None = None
        self.feature_importances_: pd.Series | None = None

    def _fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        meta: Optional[pd.DataFrame] = None,
    ) -> None:
        X_fit = X.copy()
        y_fit = y.copy()

        self.fill_values_ = X_fit.median(numeric_only=True)
        self.fill_values_ = self.fill_values_.reindex(X_fit.columns).fillna(0.0)
        X_fit = X_fit.fillna(self.fill_values_)

        self.model.fit(X_fit, y_fit)

        pred_in = pd.Series(self.model.predict(X_fit), index=y_fit.index)
        resid = y_fit - pred_in

        self.global_residual_std_ = float(resid.std(ddof=1)) if len(resid) > 1 else 0.0

        if meta is not None and "asset" in meta.columns:
            tmp = pd.DataFrame(
                {
                    "asset": meta["asset"].reset_index(drop=True),
                    "resid": resid.reset_index(drop=True),
                }
            )
            self.asset_residual_std_ = tmp.groupby("asset")["resid"].std(ddof=1).astype(float)
        else:
            self.asset_residual_std_ = None

        self.feature_importances_ = pd.Series(
            self.model.feature_importances_,
            index=X_fit.columns,
            name="importance",
        ).sort_values(ascending=False)

    def _predict(
        self,
        X: pd.DataFrame,
        meta: Optional[pd.DataFrame] = None,
    ) -> pd.Series | np.ndarray:
        if self.fill_values_ is None:
            raise ValueError("fill_values_가 없습니다. fit 후 사용하세요.")

        X_pred = X.copy()
        X_pred = X_pred.fillna(self.fill_values_)
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

        return np.full(len(X), self.global_residual_std_, dtype=float)