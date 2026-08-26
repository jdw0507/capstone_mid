from __future__ import annotations

import pandas as pd

from .base import RiskModel


class SampleCovariance(RiskModel):
    """
    표본 공분산(sample covariance) 기반 리스크 모델.

    예:
    - 월수익률 데이터면 periods_per_year=12
    - 일수익률 데이터면 periods_per_year=252
    """

    def __init__(
        self,
        periods_per_year: int = 12,
        min_obs: int = 2,
    ) -> None:
        self.periods_per_year = periods_per_year
        self.min_obs = min_obs

    def fit_predict(self, returns: pd.DataFrame) -> pd.DataFrame:
        """
        표본 공분산을 연율화해서 반환.

        Parameters
        ----------
        returns : pd.DataFrame
            index=date, columns=ticker 인 수익률 데이터

        Returns
        -------
        pd.DataFrame
            연율화된 공분산 행렬
        """
        self._validate_input(returns)

        valid_counts = returns.notna().sum()
        eligible_cols = valid_counts[valid_counts >= self.min_obs].index.tolist()

        if len(eligible_cols) == 0:
            raise ValueError("min_obs 조건을 만족하는 자산이 없습니다.")

        filtered_returns = returns[eligible_cols].copy()

        cov = filtered_returns.cov() * self.periods_per_year
        cov = cov.sort_index(axis=0).sort_index(axis=1)

        return cov

    def _validate_input(self, returns: pd.DataFrame) -> None:
        if not isinstance(returns, pd.DataFrame):
            raise TypeError("returns는 pandas DataFrame이어야 합니다.")

        if returns.empty:
            raise ValueError("returns가 비어 있습니다.")

        if returns.shape[1] == 0:
            raise ValueError("returns에 자산 열이 없습니다.")