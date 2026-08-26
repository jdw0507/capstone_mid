from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional, Union, Dict

import pandas as pd
import yfinance as yf
import numpy as np

@dataclass
class PriceLoadResult:
    prices: pd.DataFrame
    raw_data: pd.DataFrame
    valid_tickers: List[str]
    failed_tickers: List[str]
    missing_summary: pd.Series


def _normalize_tickers(tickers: Union[str, Iterable[str]]) -> List[str]:
    """
    입력 티커를 정리해서 중복 제거 후 리스트로 반환.
    """
    if isinstance(tickers, str):
        tickers = [tickers]

    cleaned = []
    seen = set()
    for t in tickers:
        if t is None:
            continue
        t = str(t).strip().upper()
        if not t:
            continue
        if t not in seen:
            cleaned.append(t)
            seen.add(t)
    return cleaned


def load_yahoo_close_prices(
    tickers: Union[str, Iterable[str]],
    start: Optional[str] = "1900-01-01",
    end: Optional[str] = None,
    price_col: str = "Adj Close",
    auto_adjust: bool = False,
    keepna: bool = False,
    repair: bool = True,
    forward_fill: bool = False,
    drop_all_nan_rows: bool = True,
    min_valid_ratio: float = 0.0,
) -> PriceLoadResult:
    """
    Yahoo Finance에서 여러 티커의 일별 종가 데이터를 가져와 DataFrame으로 반환.

    Parameters
    ----------
    tickers : str | Iterable[str]
        예: "SPY" 또는 ["SPY", "TLT", "GLD"]
    start : str, optional
        시작일. 최대한 길게 가져오려면 아주 이른 날짜로 두면 됨.
    end : str, optional
        종료일
    price_col : str
        기본은 "Adj Close". 없으면 "Close"로 fallback
    auto_adjust : bool
        False 권장. 최근 yfinance 동작 변화 때문에 명시적으로 두는 편이 안전.
    keepna : bool
        Yahoo 원본 결측 행 유지 여부
    repair : bool
        yfinance의 가격 이상치(통화 단위 오류 등) 보정 시도 옵션
    forward_fill : bool
        결측치를 이전 값으로 채울지 여부
    drop_all_nan_rows : bool
        모든 자산이 NaN인 날짜를 제거할지 여부
    min_valid_ratio : float
        열별로 유효값 비율이 이 값보다 작은 티커는 제거.
        예: 0.8이면 전체 기간 중 80% 이상 값이 있어야 유지.

    Returns
    -------
    PriceLoadResult
    """
    tickers = _normalize_tickers(tickers)
    if not tickers:
        raise ValueError("유효한 ticker가 없습니다.")

    raw = yf.download(
        tickers=tickers,
        start=start,
        end=end,
        interval="1d",
        auto_adjust=auto_adjust,
        actions=False,
        repair=repair,
        keepna=keepna,
        progress=False,
        group_by="column",
        threads=True,
    )

    if raw.empty:
        raise ValueError("Yahoo Finance에서 데이터를 가져오지 못했습니다.")

    # 단일 티커 / 다중 티커 케이스 처리
    prices = _extract_price_frame(raw=raw, tickers=tickers, preferred_col=price_col)

    # index 정리
    prices.index = pd.to_datetime(prices.index)
    prices = prices.sort_index()
    prices = prices[~prices.index.duplicated(keep="first")]

    # 숫자형 강제 변환
    prices = prices.apply(pd.to_numeric, errors="coerce")

    if forward_fill:
        prices = prices.ffill()

    if drop_all_nan_rows:
        prices = prices.dropna(how="all")

    # 유효값 비율 기준 필터링
    if min_valid_ratio > 0:
        valid_ratio = prices.notna().mean()
        keep_cols = valid_ratio[valid_ratio >= min_valid_ratio].index.tolist()
        prices = prices[keep_cols]

    # 티커 성공/실패 판별
    valid_tickers = prices.columns[prices.notna().sum() > 0].tolist()
    failed_tickers = [t for t in tickers if t not in valid_tickers]

    missing_summary = prices.isna().sum().sort_values(ascending=False)

    return PriceLoadResult(
        prices=prices,
        raw_data=raw,
        valid_tickers=valid_tickers,
        failed_tickers=failed_tickers,
        missing_summary=missing_summary,
    )


def _extract_price_frame(raw: pd.DataFrame, tickers: List[str], preferred_col: str) -> pd.DataFrame:
    """
    yfinance raw 결과에서 가격 컬럼(Adj Close 또는 Close)을 뽑아
    columns=ticker 형태의 DataFrame으로 정리.
    """
    # 다중 티커: MultiIndex columns
    if isinstance(raw.columns, pd.MultiIndex):
        level0 = raw.columns.get_level_values(0)

        if preferred_col in level0:
            prices = raw[preferred_col].copy()
        elif "Close" in level0:
            prices = raw["Close"].copy()
        else:
            available = sorted(set(level0))
            raise ValueError(
                f"원하는 가격 컬럼을 찾지 못했습니다. 사용 가능한 컬럼: {available}"
            )

        # 혹시 단일 ticker인데 Series로 나오는 경우 방지
        if isinstance(prices, pd.Series):
            prices = prices.to_frame(name=tickers[0])

        # 컬럼 순서 티커 입력 순서에 맞춤
        existing_cols = [c for c in tickers if c in prices.columns]
        prices = prices.reindex(columns=existing_cols)
        return prices

    # 단일 티커: 일반 columns
    if preferred_col in raw.columns:
        series = raw[preferred_col].copy()
    elif "Close" in raw.columns:
        series = raw["Close"].copy()
    else:
        raise ValueError(
            f"원하는 가격 컬럼을 찾지 못했습니다. columns={list(raw.columns)}"
        )

    return series.to_frame(name=tickers[0])


def compute_simple_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """
    단순 수익률 계산.
    """
    return prices.pct_change().dropna(how="all")


def compute_log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """
    로그 수익률 계산.
    """
    log_returns = np.log(prices / prices.shift(1))
    return log_returns.dropna(how="all")

def compute_log_returns_vectorized(prices: pd.DataFrame) -> pd.DataFrame:
    """
    로그 수익률 계산 (벡터화 버전).
    """
    import numpy as np
    return np.log(prices / prices.shift(1)).dropna(how="all")


def summarize_price_data(prices: pd.DataFrame) -> pd.DataFrame:
    """
    데이터 품질 및 기간 요약 테이블.
    """
    summary = pd.DataFrame({
        "start_date": prices.apply(lambda col: col.dropna().index.min()),
        "end_date": prices.apply(lambda col: col.dropna().index.max()),
        "n_obs": prices.notna().sum(),
        "missing_count": prices.isna().sum(),
        "missing_ratio": prices.isna().mean(),
    })
    return summary.sort_index()