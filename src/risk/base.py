from __future__ import annotations

from abc import ABC, abstractmethod
import pandas as pd


class RiskModel(ABC):
    """
    리스크 모델의 공통 인터페이스.
    모든 모델은 returns(DataFrame)을 받아
    asset x asset 공분산 행렬(DataFrame)을 반환한다.
    """

    @abstractmethod
    def fit_predict(self, returns: pd.DataFrame) -> pd.DataFrame:
        """
        returns: index=date, columns=ticker 인 수익률 데이터
        return: index/columns=ticker 인 공분산 행렬
        """
        raise NotImplementedError