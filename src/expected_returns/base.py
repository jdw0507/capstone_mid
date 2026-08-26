from __future__ import annotations

from abc import ABC, abstractmethod
import pandas as pd


class ExpectedReturnModel(ABC):
    """
    기대수익률 모델의 공통 인터페이스.
    모든 모델은 returns(DataFrame)을 받아
    asset별 expected return(Series)을 반환한다.
    """

    @abstractmethod
    def fit_predict(self, returns: pd.DataFrame) -> pd.Series:
        """
        returns: index=date, columns=ticker 인 수익률 데이터
        return: index=ticker 인 기대수익률 Series
        """
        raise NotImplementedError