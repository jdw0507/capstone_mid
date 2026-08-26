from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd


@dataclass
class ForecastResult:
    """
    예측 결과 컨테이너
    """
    prediction: pd.Series
    uncertainty: pd.Series | None = None
    meta: pd.DataFrame | None = None


class BaseForecaster(ABC):
    """
    모든 forecasting model의 공통 인터페이스

    기본 사용 예시
    --------------
    model.fit(X_train, y_train, meta_train)
    pred = model.predict(X_test, meta_test)
    unc  = model.predict_uncertainty(X_test, meta_test)
    result = model.predict_frame(X_test, meta_test)
    """

    def __init__(
        self,
        model_name: Optional[str] = None,
        feature_columns: Optional[list[str]] = None,
        target_column: str = "target_5d",
        horizon_days: int = 5,
        periods_per_year: int = 252,
    ) -> None:
        self.model_name = model_name or self.__class__.__name__
        self.feature_columns = feature_columns
        self.target_column = target_column
        self.horizon_days = horizon_days
        self.periods_per_year = periods_per_year

        self.is_fitted_: bool = False
        self.n_train_samples_: int | None = None
        self.fitted_feature_columns_: list[str] | None = None

    # -----------------------------
    # Public API
    # -----------------------------
    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        meta: Optional[pd.DataFrame] = None,
    ) -> "BaseForecaster":
        X, y, meta = self._validate_fit_inputs(X, y, meta)

        self._fit(X, y, meta)

        self.is_fitted_ = True
        self.n_train_samples_ = len(X)
        self.fitted_feature_columns_ = list(X.columns)

        return self

    def predict(
        self,
        X: pd.DataFrame,
        meta: Optional[pd.DataFrame] = None,
    ) -> pd.Series:
        self._check_is_fitted()
        X, meta = self._validate_predict_inputs(X, meta)

        pred = self._predict(X, meta)
        pred = self._to_prediction_series(pred, X=X, meta=meta, name="prediction")

        return pred

    def predict_uncertainty(
        self,
        X: pd.DataFrame,
        meta: Optional[pd.DataFrame] = None,
    ) -> pd.Series | None:
        """
        기본값은 None
        필요하면 하위 클래스에서 override
        """
        self._check_is_fitted()
        X, meta = self._validate_predict_inputs(X, meta)

        unc = self._predict_uncertainty(X, meta)
        if unc is None:
            return None

        unc = self._to_prediction_series(unc, X=X, meta=meta, name="uncertainty")
        return unc

    def fit_predict(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: pd.DataFrame,
        meta_train: Optional[pd.DataFrame] = None,
        meta_test: Optional[pd.DataFrame] = None,
    ) -> pd.Series:
        self.fit(X_train, y_train, meta_train)
        return self.predict(X_test, meta_test)

    def predict_frame(
        self,
        X: pd.DataFrame,
        meta: Optional[pd.DataFrame] = None,
    ) -> ForecastResult:
        pred = self.predict(X, meta)
        unc = self.predict_uncertainty(X, meta)

        meta_out = None
        if meta is not None:
            meta_out = meta.copy().reset_index(drop=True)

        return ForecastResult(
            prediction=pred,
            uncertainty=unc,
            meta=meta_out,
        )

    # -----------------------------
    # Methods subclasses must implement
    # -----------------------------
    @abstractmethod
    def _fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        meta: Optional[pd.DataFrame] = None,
    ) -> None:
        pass

    @abstractmethod
    def _predict(
        self,
        X: pd.DataFrame,
        meta: Optional[pd.DataFrame] = None,
    ) -> pd.Series | np.ndarray:
        pass

    def _predict_uncertainty(
        self,
        X: pd.DataFrame,
        meta: Optional[pd.DataFrame] = None,
    ) -> pd.Series | np.ndarray | None:
        return None

    # -----------------------------
    # Validation helpers
    # -----------------------------
    def _validate_fit_inputs(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        meta: Optional[pd.DataFrame],
    ) -> tuple[pd.DataFrame, pd.Series, Optional[pd.DataFrame]]:
        if not isinstance(X, pd.DataFrame):
            raise TypeError("X는 pandas DataFrame이어야 합니다.")
        if not isinstance(y, pd.Series):
            raise TypeError("y는 pandas Series이어야 합니다.")

        if len(X) != len(y):
            raise ValueError(f"X와 y 길이가 다릅니다. len(X)={len(X)}, len(y)={len(y)}")

        if meta is not None and len(meta) != len(X):
            raise ValueError(f"meta와 X 길이가 다릅니다. len(meta)={len(meta)}, len(X)={len(X)}")

        X = X.copy()
        y = y.copy()

        X = X.apply(pd.to_numeric, errors="coerce")
        y = pd.to_numeric(y, errors="coerce")

        if self.feature_columns is not None:
            missing = [c for c in self.feature_columns if c not in X.columns]
            if missing:
                raise ValueError(f"X에 필요한 feature_columns가 없습니다: {missing}")
            X = X[self.feature_columns].copy()

        # fit 시에는 target missing 제거
        valid_mask = y.notna()
        X = X.loc[valid_mask].reset_index(drop=True)
        y = y.loc[valid_mask].reset_index(drop=True)

        if meta is not None:
            meta = meta.loc[valid_mask].reset_index(drop=True)

        if X.empty or y.empty:
            raise ValueError("유효한 학습 데이터가 없습니다.")

        return X, y, meta

    def _validate_predict_inputs(
        self,
        X: pd.DataFrame,
        meta: Optional[pd.DataFrame],
    ) -> tuple[pd.DataFrame, Optional[pd.DataFrame]]:
        if not isinstance(X, pd.DataFrame):
            raise TypeError("X는 pandas DataFrame이어야 합니다.")

        X = X.copy()
        X = X.apply(pd.to_numeric, errors="coerce")

        if self.fitted_feature_columns_ is not None:
            missing = [c for c in self.fitted_feature_columns_ if c not in X.columns]
            if missing:
                raise ValueError(f"예측에 필요한 feature가 없습니다: {missing}")
            X = X[self.fitted_feature_columns_].copy()

        if meta is not None:
            if len(meta) != len(X):
                raise ValueError(f"meta와 X 길이가 다릅니다. len(meta)={len(meta)}, len(X)={len(X)}")
            meta = meta.copy().reset_index(drop=True)

        X = X.reset_index(drop=True)

        if X.empty:
            raise ValueError("예측용 X가 비어 있습니다.")

        return X, meta

    def _check_is_fitted(self) -> None:
        if not self.is_fitted_:
            raise ValueError(f"{self.model_name} 모델이 아직 fit되지 않았습니다.")

    def _to_prediction_series(
        self,
        values: pd.Series | np.ndarray,
        X: pd.DataFrame,
        meta: Optional[pd.DataFrame],
        name: str,
    ) -> pd.Series:
        if isinstance(values, pd.Series):
            out = values.copy().reset_index(drop=True)
        else:
            arr = np.asarray(values).reshape(-1)
            if len(arr) != len(X):
                raise ValueError(
                    f"예측 길이가 입력 길이와 다릅니다. len(pred)={len(arr)}, len(X)={len(X)}"
                )
            out = pd.Series(arr)

        out.name = name

        # meta가 있으면 MultiIndex 형태로 주는 게 나중에 편함
        if meta is not None and {"date", "asset"}.issubset(meta.columns):
            idx = pd.MultiIndex.from_frame(meta[["date", "asset"]])
            out.index = idx

        return out