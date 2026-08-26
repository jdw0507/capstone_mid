from __future__ import annotations

from typing import Optional, Literal
import numpy as np
import pandas as pd


FillMethod = Literal["none", "ffill", "bfill", "both"]


def ensure_datetime_index(df: pd.DataFrame) -> pd.DataFrame:
    """
    인덱스를 DatetimeIndex로 변환하고 정렬/중복 제거.
    """
    out = df.copy()
    out.index = pd.to_datetime(out.index)
    out = out.sort_index()
    out = out[~out.index.duplicated(keep="first")]
    return out


def clean_price_data(
    prices: pd.DataFrame,
    fill_method: FillMethod = "none",
    drop_all_nan_rows: bool = True,
) -> pd.DataFrame:
    """
    가격 데이터 기본 정리:
    - datetime index 보장
    - 숫자형 변환
    - fill 처리
    - 전체 NaN 행 제거
    """
    out = ensure_datetime_index(prices)
    out = out.apply(pd.to_numeric, errors="coerce")

    if fill_method == "ffill":
        out = out.ffill()
    elif fill_method == "bfill":
        out = out.bfill()
    elif fill_method == "both":
        out = out.ffill().bfill()
    elif fill_method == "none":
        pass
    else:
        raise ValueError(f"지원하지 않는 fill_method: {fill_method}")

    if drop_all_nan_rows:
        out = out.dropna(how="all")

    return out


def filter_assets_by_valid_ratio(
    prices: pd.DataFrame,
    min_valid_ratio: float = 0.8,
) -> pd.DataFrame:
    """
    유효값 비율이 min_valid_ratio 이상인 자산만 남김.
    """
    if not 0 <= min_valid_ratio <= 1:
        raise ValueError("min_valid_ratio는 0~1 사이여야 합니다.")

    valid_ratio = prices.notna().mean()
    keep_cols = valid_ratio[valid_ratio >= min_valid_ratio].index.tolist()
    return prices[keep_cols].copy()


def subset_by_date(
    prices: pd.DataFrame,
    start: Optional[str] = None,
    end: Optional[str] = None,
) -> pd.DataFrame:
    """
    날짜 범위로 자르기.
    """
    out = prices.copy()
    if start is not None:
        out = out.loc[pd.to_datetime(start):]
    if end is not None:
        out = out.loc[:pd.to_datetime(end)]
    return out


def align_to_common_period(
    prices: pd.DataFrame,
    drop_remaining_nan: bool = True,
) -> pd.DataFrame:
    """
    모든 자산이 동시에 존재하는 공통 구간만 남김.

    예:
    SPY는 1993, QQQ는 1999, GLD는 2004부터 있으면
    공통 구간은 대체로 2004 이후가 됨.
    """
    out = ensure_datetime_index(prices)

    first_valid_dates = out.apply(lambda col: col.first_valid_index())
    if first_valid_dates.isna().any():
        valid_cols = first_valid_dates[first_valid_dates.notna()].index.tolist()
        out = out[valid_cols]
        first_valid_dates = out.apply(lambda col: col.first_valid_index())

    common_start = first_valid_dates.max()
    out = out.loc[common_start:]

    if drop_remaining_nan:
        out = out.dropna(how="any")

    return out


def drop_assets_with_any_nan(prices: pd.DataFrame) -> pd.DataFrame:
    """
    NaN이 하나라도 있는 자산(열)을 제거.
    보수적으로 완전한 자산만 남기고 싶을 때 사용.
    """
    return prices.dropna(axis=1, how="any").copy()


def resample_prices(
    prices: pd.DataFrame,
    freq: str = "M",
) -> pd.DataFrame:
    """
    가격 데이터를 지정 주기 말단 가격으로 리샘플링.

    freq 예시:
    - 'D'  : 일별
    - 'W'  : 주별
    - 'M'  : 월말
    - 'ME' : 월말(일부 환경에서 권장)
    - 'Q'  : 분기말
    - 'Y'  : 연말
    """
    out = ensure_datetime_index(prices)
    out = out.resample(freq).last()
    out = out.dropna(how="all")
    return out


def to_month_end_prices(prices: pd.DataFrame) -> pd.DataFrame:
    """
    월말 가격으로 변환.
    """
    return resample_prices(prices, freq="ME")


def to_week_end_prices(prices: pd.DataFrame) -> pd.DataFrame:
    """
    주말 가격으로 변환.
    """
    return resample_prices(prices, freq="W")

def annualize_returns(returns: pd.DataFrame, periods_per_year: int = 12) -> pd.Series:
    """
    평균 수익률 연율화
    """
    return returns.mean() * periods_per_year

def annualize_covariance(returns: pd.DataFrame, periods_per_year: int = 12) -> pd.DataFrame:
    """
    공분산 행렬 연율화
    """
    return returns.cov() * periods_per_year

def compute_cumulative_returns(returns: pd.DataFrame) -> pd.DataFrame:
    """
    누적수익률 계산
    """
    return (1 + returns).cumprod()

def compute_simple_returns(
    prices: pd.DataFrame,
    dropna: bool = True,
) -> pd.DataFrame:
    """
    단순 수익률 계산.
    """
    returns = prices.pct_change()
    if dropna:
        returns = returns.dropna(how="all")
    return returns


def compute_log_returns(
    prices: pd.DataFrame,
    dropna: bool = True,
) -> pd.DataFrame:
    """
    로그 수익률 계산.
    """
    safe_prices = prices.where(prices > 0)
    returns = np.log(safe_prices / safe_prices.shift(1))
    if dropna:
        returns = returns.dropna(how="all")
    return returns


def winsorize_returns(
    returns: pd.DataFrame,
    lower: float = 0.01,
    upper: float = 0.99,
) -> pd.DataFrame:
    """
    극단값 완화용 winsorization.
    자산별로 분위수 기준 clip 적용.
    """
    out = returns.copy()
    lower_q = out.quantile(lower)
    upper_q = out.quantile(upper)
    out = out.clip(lower=lower_q, upper=upper_q, axis=1)
    return out


def validate_price_data(
    prices: pd.DataFrame,
    min_rows: int = 30,
    min_cols: int = 1,
) -> None:
    """
    기본 검증.
    """
    if prices.empty:
        raise ValueError("가격 데이터가 비어 있습니다.")

    if prices.shape[0] < min_rows:
        raise ValueError(f"행 수가 너무 적습니다. 현재 {prices.shape[0]}행")

    if prices.shape[1] < min_cols:
        raise ValueError(f"열 수가 너무 적습니다. 현재 {prices.shape[1]}열")

    if not isinstance(prices.index, pd.DatetimeIndex):
        raise TypeError("index는 DatetimeIndex여야 합니다.")


def summarize_price_data(prices: pd.DataFrame) -> pd.DataFrame:
    """
    자산별 가격 데이터 요약표.
    """
    summary = pd.DataFrame({
        "start_date": prices.apply(lambda col: col.dropna().index.min()),
        "end_date": prices.apply(lambda col: col.dropna().index.max()),
        "n_obs": prices.notna().sum(),
        "missing_count": prices.isna().sum(),
        "missing_ratio": prices.isna().mean(),
    })
    return summary.sort_index()


def summarize_return_data(returns: pd.DataFrame) -> pd.DataFrame:
    """
    자산별 수익률 데이터 요약표.
    """
    summary = pd.DataFrame({
        "mean": returns.mean(),
        "std": returns.std(),
        "min": returns.min(),
        "max": returns.max(),
        "missing_count": returns.isna().sum(),
        "missing_ratio": returns.isna().mean(),
    })
    return summary.sort_index()


def prepare_price_data_for_allocation(
    prices: pd.DataFrame,
    start: Optional[str] = None,
    end: Optional[str] = None,
    fill_method: FillMethod = "none",
    use_common_period: bool = True,
    resample_freq: Optional[str] = None,
) -> pd.DataFrame:
    """
    자산배분용 가격 데이터 준비용 통합 함수.

    순서:
    1. 기본 정리
    2. 날짜 자르기
    3. 공통 구간 정렬
    4. 리샘플링
    """
    out = clean_price_data(prices, fill_method=fill_method, drop_all_nan_rows=True)
    out = subset_by_date(out, start=start, end=end)

    if use_common_period:
        out = align_to_common_period(out, drop_remaining_nan=True)

    if resample_freq is not None:
        out = resample_prices(out, freq=resample_freq)

    validate_price_data(out)
    return out