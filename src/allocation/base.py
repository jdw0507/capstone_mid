from __future__ import annotations

from abc import ABC, abstractmethod
import pandas as pd


class AllocationModel(ABC):
    """
    자산배분 모델 공통 인터페이스.
    mu(기대수익률), cov(공분산)을 받아
    portfolio weights(Series)를 반환한다.
    """

    @abstractmethod
    def optimize(self, expected_returns: pd.Series, covariance: pd.DataFrame) -> pd.Series:
        raise NotImplementedError