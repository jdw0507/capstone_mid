from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

from src.forecasting.base import BaseForecaster


class RandomForestForecaster(BaseForecaster):
    """
    Random Forest baseline forecaster
    """

    def __init__(
        self,
        n_estimators: int = 400,
        max_depth: int | None = 8,
        min_samples_leaf: int = 10,
        max_features: str | float | int = "sqrt",
        n_jobs: int = 1,
        random_state: int = 42,
        model_name: Optional[str] = None,
        feature_columns: Optional[list[str]] = None,
        target_column: str = "target_5d_excess",
        horizon_days: int = 5,
        periods_per_year: int = 252,
    ) -> None:
        super().__init__(
            model_name=model_name or "RandomForest",
            feature_columns=feature_columns,
            target_column=target_column,
            horizon_days=horizon_days,
            periods_per_year=periods_per_year,
        )

        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.n_jobs = n_jobs
        self.random_state = random_state

        self.model = RandomForestRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            min_samples_leaf=self.min_samples_leaf,
            max_features=self.max_features,
            random_state=self.random_state,
            n_jobs=self.n_jobs,
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
        X_pred = X.copy().fillna(self.fill_values_)
        return self.model.predict(X_pred)

    def _predict_uncertainty(
        self,
        X: pd.DataFrame,
        meta: Optional[pd.DataFrame] = None,
    ) -> pd.Series | np.ndarray | None:
        if self.fill_values_ is None:
            return None

        X_pred = X.copy().fillna(self.fill_values_).to_numpy()
        tree_preds = np.column_stack([tree.predict(X_pred) for tree in self.model.estimators_])
        unc = tree_preds.std(axis=1, ddof=1)
        return unc
