from __future__ import annotations

from copy import deepcopy
from typing import Optional, Literal

import numpy as np
import pandas as pd

from src.forecasting.base import BaseForecaster


EnsembleMethod = Literal["mean", "median", "weighted_mean"]


class EnsembleForecaster(BaseForecaster):
    """
    여러 forecaster를 묶는 ensemble model

    Parameters
    ----------
    forecasters : dict[str, BaseForecaster]
        개별 예측 모델 딕셔너리
    method : {"mean", "median", "weighted_mean"}
        앙상블 방식
    weights : dict[str, float] | None
        weighted_mean일 때 사용할 가중치
    """

    def __init__(
        self,
        forecasters: dict[str, BaseForecaster],
        method: EnsembleMethod = "mean",
        weights: Optional[dict[str, float]] = None,
        model_name: Optional[str] = None,
        feature_columns: Optional[list[str]] = None,
        target_column: str = "target_5d_excess",
        horizon_days: int = 5,
        periods_per_year: int = 252,
    ) -> None:
        super().__init__(
            model_name=model_name or f"Ensemble_{method}",
            feature_columns=feature_columns,
            target_column=target_column,
            horizon_days=horizon_days,
            periods_per_year=periods_per_year,
        )

        if len(forecasters) == 0:
            raise ValueError("forecasters는 비어 있을 수 없습니다.")

        self.forecasters = forecasters
        self.method = method
        self.weights = weights

        self.fitted_models_: dict[str, BaseForecaster] = {}

    def _fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        meta: Optional[pd.DataFrame] = None,
    ) -> None:
        self.fitted_models_ = {}
        for name, model in self.forecasters.items():
            mdl = deepcopy(model)
            mdl.fit(X, y, meta)
            self.fitted_models_[name] = mdl

    def _predict(
        self,
        X: pd.DataFrame,
        meta: Optional[pd.DataFrame] = None,
    ) -> pd.Series | np.ndarray:
        preds = {}
        for name, mdl in self.fitted_models_.items():
            preds[name] = mdl.predict(X, meta)

        pred_df = pd.concat(
            [pd.Series(v).reset_index(drop=True).rename(k) for k, v in preds.items()],
            axis=1,
        )

        if self.method == "mean":
            out = pred_df.mean(axis=1)

        elif self.method == "median":
            out = pred_df.median(axis=1)

        elif self.method == "weighted_mean":
            if self.weights is None:
                raise ValueError("weighted_mean을 쓰려면 weights가 필요합니다.")

            w = pd.Series(self.weights, dtype=float)
            w = w.reindex(pred_df.columns).fillna(0.0)
            if np.isclose(w.sum(), 0.0):
                raise ValueError("weights 합이 0입니다.")
            w = w / w.sum()

            out = pred_df.mul(w, axis=1).sum(axis=1)

        else:
            raise ValueError(f"지원하지 않는 ensemble method: {self.method}")

        return out.values

    def _predict_uncertainty(
        self,
        X: pd.DataFrame,
        meta: Optional[pd.DataFrame] = None,
    ) -> pd.Series | np.ndarray | None:
        # 모델 간 disagreement를 uncertainty proxy로 사용
        preds = {}
        for name, mdl in self.fitted_models_.items():
            preds[name] = mdl.predict(X, meta)

        pred_df = pd.concat(
            [pd.Series(v).reset_index(drop=True).rename(k) for k, v in preds.items()],
            axis=1,
        )

        unc = pred_df.std(axis=1, ddof=1).fillna(0.0)
        return unc.values