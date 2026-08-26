from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Union

import pandas as pd


@dataclass(frozen=True)
class NestedWalkForwardConfig:
    model_train_days: int = 1008   # 약 4년
    view_build_days: int = 252     # 약 1년
    test_days: int = 756           # 약 3년
    step_days: int = 63            # 약 1분기


def prepare_trading_index(
    data: Union[pd.DataFrame, pd.Series, pd.Index],
    date_col: str = "date",
) -> pd.DatetimeIndex:
    """
    입력 데이터를 정렬된 고유 DatetimeIndex로 변환
    """
    if isinstance(data, pd.Index):
        idx = pd.to_datetime(data)
    elif isinstance(data, pd.Series):
        idx = pd.to_datetime(data.index)
    elif isinstance(data, pd.DataFrame):
        if isinstance(data.index, pd.DatetimeIndex):
            idx = pd.to_datetime(data.index)
        elif date_col in data.columns:
            idx = pd.to_datetime(data[date_col])
        else:
            raise ValueError("DatetimeIndex 또는 date_col이 필요합니다.")
    else:
        raise TypeError("data는 DataFrame / Series / Index 중 하나여야 합니다.")

    idx = pd.DatetimeIndex(sorted(pd.Index(idx).dropna().unique()))
    if len(idx) == 0:
        raise ValueError("유효한 날짜 인덱스가 없습니다.")

    return idx


def build_nested_walk_forward_splits(
    data: Union[pd.DataFrame, pd.Series, pd.Index],
    model_train_days: int = 1008,
    view_build_days: int = 252,
    test_days: int = 756,
    step_days: int = 63,
    date_col: str = "date",
) -> pd.DataFrame:
    """
    Nested walk-forward split 생성

    Fold 구조
    ---------
    [ model_train | view_build | test ]

    예:
    - model_train_days = 1008 (약 4년)
    - view_build_days = 252  (약 1년)
    - test_days = 756        (약 3년)
    - step_days = 63         (약 1분기)

    Returns
    -------
    pd.DataFrame
        columns:
        - fold_id
        - model_train_start
        - model_train_end
        - view_start
        - view_end
        - test_start
        - test_end
        - n_model_train
        - n_view_build
        - n_test
    """
    idx = prepare_trading_index(data, date_col=date_col)

    total_needed = model_train_days + view_build_days + test_days
    if len(idx) < total_needed:
        raise ValueError(
            f"데이터가 부족합니다. 최소 {total_needed} 거래일 필요, 현재 {len(idx)} 거래일."
        )

    rows = []
    fold_id = 0
    start_pos = 0

    while True:
        model_train_start_pos = start_pos
        model_train_end_pos = model_train_start_pos + model_train_days - 1

        view_start_pos = model_train_end_pos + 1
        view_end_pos = view_start_pos + view_build_days - 1

        test_start_pos = view_end_pos + 1
        test_end_pos = test_start_pos + test_days - 1

        if test_end_pos >= len(idx):
            break

        rows.append(
            {
                "fold_id": fold_id,
                "model_train_start": idx[model_train_start_pos],
                "model_train_end": idx[model_train_end_pos],
                "view_start": idx[view_start_pos],
                "view_end": idx[view_end_pos],
                "test_start": idx[test_start_pos],
                "test_end": idx[test_end_pos],
                "n_model_train": model_train_days,
                "n_view_build": view_build_days,
                "n_test": test_days,
            }
        )

        fold_id += 1
        start_pos += step_days

    if len(rows) == 0:
        raise ValueError("생성된 walk-forward fold가 없습니다.")

    return pd.DataFrame(rows)


def slice_by_split(
    data: pd.DataFrame,
    split_row: pd.Series,
    date_col: str = "date",
) -> dict[str, pd.DataFrame]:
    """
    하나의 split row를 기준으로 data를
    model_train / view_build / test 로 잘라 반환
    """
    if isinstance(data.index, pd.DatetimeIndex):
        df = data.copy()
    else:
        if date_col not in data.columns:
            raise ValueError("DatetimeIndex 또는 date_col이 필요합니다.")
        df = data.copy()
        df[date_col] = pd.to_datetime(df[date_col])

    def _mask(start, end):
        if isinstance(df.index, pd.DatetimeIndex):
            return (df.index >= start) & (df.index <= end)
        return (df[date_col] >= start) & (df[date_col] <= end)

    out = {
        "model_train": df.loc[_mask(split_row["model_train_start"], split_row["model_train_end"])].copy(),
        "view_build": df.loc[_mask(split_row["view_start"], split_row["view_end"])].copy(),
        "test": df.loc[_mask(split_row["test_start"], split_row["test_end"])].copy(),
    }
    return out


def build_prediction_schedule(
    data: Union[pd.DataFrame, pd.Index],
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    every_n_days: int = 5,
    forecast_horizon_days: int = 5,
    date_col: str = "date",
) -> pd.DataFrame:
    """
    테스트 구간 내부에서 예측/리밸런싱 스케줄 생성

    예:
    - 매 5거래일마다 예측/리밸런싱
    - 다음 5거래일 target을 사용

    Returns
    -------
    pd.DataFrame
        columns:
        - step_id
        - decision_date
        - forecast_end_date
        - next_decision_date
    """
    idx = prepare_trading_index(data, date_col=date_col)

    test_idx = idx[(idx >= pd.Timestamp(start_date)) & (idx <= pd.Timestamp(end_date))]
    if len(test_idx) <= forecast_horizon_days:
        raise ValueError("테스트 구간이 너무 짧습니다.")

    rows = []
    step_id = 0

    pos = 0
    while True:
        decision_pos = pos
        forecast_end_pos = decision_pos + forecast_horizon_days
        next_decision_pos = decision_pos + every_n_days

        if forecast_end_pos >= len(test_idx):
            break

        next_decision_date = (
            test_idx[next_decision_pos] if next_decision_pos < len(test_idx) else pd.NaT
        )

        rows.append(
            {
                "step_id": step_id,
                "decision_date": test_idx[decision_pos],
                "forecast_end_date": test_idx[forecast_end_pos],
                "next_decision_date": next_decision_date,
            }
        )

        step_id += 1
        pos += every_n_days

    return pd.DataFrame(rows)


def describe_split_coverage(splits: pd.DataFrame) -> pd.DataFrame:
    """
    split DataFrame 요약 정보
    """
    if splits.empty:
        raise ValueError("splits가 비어 있습니다.")

    out = pd.DataFrame(
        {
            "n_folds": [len(splits)],
            "first_model_train_start": [splits["model_train_start"].min()],
            "last_test_end": [splits["test_end"].max()],
            "avg_model_train_days": [splits["n_model_train"].mean()],
            "avg_view_build_days": [splits["n_view_build"].mean()],
            "avg_test_days": [splits["n_test"].mean()],
        }
    )
    return out