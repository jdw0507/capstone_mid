from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from src.forecasting.base import BaseForecaster

try:
    from statsmodels.tsa.arima.model import ARIMA
except ImportError:  # pragma: no cover
    ARIMA = None


class ARIMAForecaster(BaseForecaster):
    """
    Asset-wise univariate ARIMA forecaster

    주의
    ----
    - X는 실제로 사용하지 않음
    - meta에 date, asset 컬럼이 있어야 함
    - 자산별 y 시계열만으로 학습/예측
    """

    def __init__(
        self,
        order: tuple[int, int, int] = (1, 0, 1),
        trend: str | None = "c",
        model_name: Optional[str] = None,
        feature_columns: Optional[list[str]] = None,
        target_column: str = "target_5d_excess",
        horizon_days: int = 5,
        periods_per_year: int = 252,
    ) -> None:
        super().__init__(
            model_name=model_name or "ARIMA",
            feature_columns=feature_columns,
            target_column=target_column,
            horizon_days=horizon_days,
            periods_per_year=periods_per_year,
        )

        if ARIMA is None:
            raise ImportError(
                "statsmodels가 설치되어 있지 않습니다. "
                "설치: C:/Anaconda/envs/torch-gpu/python.exe -m pip install statsmodels"
            )

        self.order = order
        self.trend = trend

        self.asset_models_: dict[str, object] = {}
        self.asset_residual_std_: pd.Series | None = None
        self.global_residual_std_: float | None = None

    def _fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        meta: Optional[pd.DataFrame] = None,
    ) -> None:
        if meta is None or not {"date", "asset"}.issubset(meta.columns):
            raise ValueError("ARIMA는 meta에 date, asset 컬럼이 필요합니다.")

        tmp = pd.DataFrame(
            {
                "date": pd.to_datetime(meta["date"]).reset_index(drop=True),
                "asset": meta["asset"].reset_index(drop=True),
                "y": y.reset_index(drop=True),
            }
        ).dropna()

        asset_std = {}
        residuals_all = []

        for asset, g in tmp.groupby("asset"):
            g = g.sort_values("date")
            series = g["y"].astype(float).reset_index(drop=True)

            if len(series) < 20:
                continue

            try:
                res = ARIMA(series,order=self.order,trend=self.trend,enforce_stationarity=False,enforce_invertibility=False,).fit()
                self.asset_models_[asset] = res

                resid = pd.Series(res.resid).dropna()
                std = float(resid.std(ddof=1)) if len(resid) > 1 else 0.0
                asset_std[asset] = std
                residuals_all.extend(resid.tolist())

            except Exception:
                continue

        if len(self.asset_models_) == 0:
            raise ValueError("학습 가능한 ARIMA 자산이 없습니다.")

        self.asset_residual_std_ = pd.Series(asset_std, name="resid_std").astype(float)

        if len(residuals_all) > 1:
            self.global_residual_std_ = float(pd.Series(residuals_all).std(ddof=1))
        else:
            self.global_residual_std_ = 0.0

    def _predict(
        self,
        X: pd.DataFrame,
        meta: Optional[pd.DataFrame] = None,
    ) -> pd.Series | np.ndarray:
        if meta is None or not {"date", "asset"}.issubset(meta.columns):
            raise ValueError("ARIMA는 meta에 date, asset 컬럼이 필요합니다.")

        meta = meta.reset_index(drop=True).copy()
        out = np.full(len(meta), np.nan, dtype=float)

        # asset별로 테스트 구간 길이만큼 multi-step forecast
        for asset, idx in meta.groupby("asset").groups.items():
            if asset not in self.asset_models_:
                continue

            locs = list(idx)
            sub = meta.loc[locs].copy().sort_values("date")
            n_steps = len(sub)

            try:
                fcst = self.asset_models_[asset].forecast(steps=n_steps)
                out[sub.index.values] = np.asarray(fcst).reshape(-1)
            except Exception:
                continue

        return out

    def _predict_uncertainty(
        self,
        X: pd.DataFrame,
        meta: Optional[pd.DataFrame] = None,
    ) -> pd.Series | np.ndarray | None:
        if meta is None or not {"date", "asset"}.issubset(meta.columns):
            if self.global_residual_std_ is None:
                return None
            return np.full(len(X), self.global_residual_std_, dtype=float)

        if self.asset_residual_std_ is None and self.global_residual_std_ is None:
            return None

        out = meta["asset"].map(self.asset_residual_std_).astype(float)
        if self.global_residual_std_ is not None:
            out = out.fillna(self.global_residual_std_)

        return out.values