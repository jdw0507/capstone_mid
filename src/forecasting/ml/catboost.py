from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from src.forecasting.base import BaseForecaster

try:
    from catboost import CatBoostRegressor
except ImportError:  # pragma: no cover
    CatBoostRegressor = None


class CatBoostForecaster(BaseForecaster):
    """
    CatBoost baseline forecaster

    특징
    ----
    - 비선형성, 상호작용 반영
    - missing은 training median으로 보정
    - uncertainty는 training residual std proxy 사용
    """

    def __init__(
        self,
        iterations: int = 500,
        learning_rate: float = 0.03,
        depth: int = 6,
        l2_leaf_reg: float = 3.0,
        random_state: int = 42,
        model_name: Optional[str] = None,
        feature_columns: Optional[list[str]] = None,
        target_column: str = "target_5d_excess",
        horizon_days: int = 5,
        periods_per_year: int = 252,
    ) -> None:
        super().__init__(
            model_name=model_name or "CatBoost",
            feature_columns=feature_columns,
            target_column=target_column,
            horizon_days=horizon_days,
            periods_per_year=periods_per_year,
        )

        if CatBoostRegressor is None:
            raise ImportError(
                "catboost가 설치되어 있지 않습니다. "
                "설치: C:/Anaconda/envs/torch-gpu/python.exe -m pip install catboost"
            )

        self.iterations = iterations
        self.learning_rate = learning_rate
        self.depth = depth
        self.l2_leaf_reg = l2_leaf_reg
        self.random_state = random_state

        self.model = CatBoostRegressor(
            iterations=self.iterations,
            learning_rate=self.learning_rate,
            depth=self.depth,
            l2_leaf_reg=self.l2_leaf_reg,
            loss_function="RMSE",
            random_seed=self.random_state,
            verbose=False,
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

        self.feature_importances_ = pd.Series(
            self.model.get_feature_importance(),
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