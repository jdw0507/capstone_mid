from __future__ import annotations

import pandas as pd


def build_rebalance_schedule(
    monthly_prices: pd.DataFrame,
    lookback_periods: int = 36,
) -> pd.DataFrame:
    """
    월말 가격 데이터 기준으로 리밸런싱 스케줄 생성.

    Parameters
    ----------
    monthly_prices : pd.DataFrame
        월말 가격 데이터. index=date, columns=ticker
    lookback_periods : int
        기대수익률/공분산 추정에 사용할 과거 수익률 개수
        예: 36이면 최근 36개월 수익률 사용

    Returns
    -------
    pd.DataFrame
        columns:
        - rebalance_date
        - next_date
        - train_start_date
        - train_end_date
    """
    prices = monthly_prices.copy()
    prices.index = pd.to_datetime(prices.index)
    prices = prices.sort_index()
    prices = prices[~prices.index.duplicated(keep="first")]

    if prices.empty:
        raise ValueError("monthly_prices가 비어 있습니다.")

    # lookback_periods 개의 returns를 만들려면 최소 lookback_periods + 1 개의 price가 필요
    # 그리고 다음 기간 수익률까지 보려면 추가로 1개가 더 필요
    min_required_rows = lookback_periods + 2
    if len(prices) < min_required_rows:
        raise ValueError(
            f"백테스트에 필요한 관측치가 부족합니다. "
            f"최소 {min_required_rows}개 월말 가격이 필요하지만 현재 {len(prices)}개입니다."
        )

    dates = prices.index
    rows = []

    for i in range(lookback_periods, len(dates) - 1):
        rebalance_date = dates[i]
        next_date = dates[i + 1]
        train_start_date = dates[i - lookback_periods]
        train_end_date = rebalance_date

        rows.append(
            {
                "rebalance_date": rebalance_date,
                "next_date": next_date,
                "train_start_date": train_start_date,
                "train_end_date": train_end_date,
            }
        )

    schedule = pd.DataFrame(rows)
    return schedule