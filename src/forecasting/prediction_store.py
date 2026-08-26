from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd


def save_forecast_artifacts(
    per_prediction: pd.DataFrame,
    summary_by_model: pd.DataFrame,
    summary_by_model_asset: pd.DataFrame,
    rank_ic_by_date: pd.DataFrame,
    output_dir: str | Path,
    run_name: str = "forecast_run",
    save_csv: bool = True,
    save_parquet: bool = True,
) -> dict[str, Path]:
    """
    예측 결과 전체를 저장

    저장 항목
    ---------
    - per_prediction : long-format 원본 예측값
    - summary_by_model
    - summary_by_model_asset
    - rank_ic_by_date

    Returns
    -------
    dict[str, Path]
        저장된 파일 경로들
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    saved = {}

    if save_parquet:
        per_pred_parquet = output_dir / f"{run_name}_per_prediction.parquet"
        per_prediction.to_parquet(per_pred_parquet, index=False)
        saved["per_prediction_parquet"] = per_pred_parquet

    if save_csv:
        per_pred_csv = output_dir / f"{run_name}_per_prediction.csv"
        per_prediction.to_csv(per_pred_csv, index=False)
        saved["per_prediction_csv"] = per_pred_csv

        summary_model_csv = output_dir / f"{run_name}_summary_by_model.csv"
        summary_by_model.to_csv(summary_model_csv)
        saved["summary_by_model_csv"] = summary_model_csv

        summary_asset_csv = output_dir / f"{run_name}_summary_by_model_asset.csv"
        summary_by_model_asset.to_csv(summary_asset_csv)
        saved["summary_by_model_asset_csv"] = summary_asset_csv

        rank_ic_csv = output_dir / f"{run_name}_rank_ic_by_date.csv"
        rank_ic_by_date.to_csv(rank_ic_csv, index=False)
        saved["rank_ic_by_date_csv"] = rank_ic_csv

    return saved


def load_saved_predictions(path: str | Path) -> pd.DataFrame:
    """
    저장된 per_prediction 파일 로드
    csv / parquet 자동 처리
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"파일이 없습니다: {path}")

    if path.suffix.lower() == ".parquet":
        df = pd.read_parquet(path)
    elif path.suffix.lower() == ".csv":
        df = pd.read_csv(path)
    else:
        raise ValueError(f"지원하지 않는 확장자입니다: {path.suffix}")

    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])

    for c in ["train_start", "train_end", "view_start", "view_end", "test_start", "test_end"]:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c])

    return df


def build_prediction_matrix(
    per_prediction: pd.DataFrame,
    model_name: str,
    value_col: str = "y_pred",
    fold_id: Optional[int] = None,
    evaluation_split: Optional[str] = None,
    dropna: bool = False,
) -> pd.DataFrame:
    """
    long-format 예측 테이블 -> date x asset wide matrix

    Parameters
    ----------
    per_prediction : pd.DataFrame
    model_name : str
    value_col : str
        예:
        - "y_pred"       -> Q 후보
        - "uncertainty"  -> Omega 후보
        - "error"        -> 오차 분석
    fold_id : int | None
        특정 fold만 보고 싶을 때
    evaluation_split : str | None
        예: "view_build", "test"
    dropna : bool
        모든 NaN 행 제거 여부
    """
    required = {"model", "date", "asset", value_col}
    missing = required - set(per_prediction.columns)
    if missing:
        raise ValueError(f"필수 컬럼이 없습니다: {missing}")

    df = per_prediction.copy()
    df["date"] = pd.to_datetime(df["date"])

    df = df[df["model"] == model_name].copy()

    if fold_id is not None:
        if "fold_id" not in df.columns:
            raise ValueError("fold_id 컬럼이 없습니다.")
        df = df[df["fold_id"] == fold_id].copy()

    if evaluation_split is not None:
        if "evaluation_split" not in df.columns:
            raise ValueError("evaluation_split 컬럼이 없습니다.")
        df = df[df["evaluation_split"] == evaluation_split].copy()

    out = df.pivot(index="date", columns="asset", values=value_col)
    out = out.sort_index().sort_index(axis=1)

    if dropna:
        out = out.dropna(how="all")

    return out


def build_q_matrix_from_predictions(
    per_prediction: pd.DataFrame,
    model_name: str,
    fold_id: Optional[int] = None,
    evaluation_split: Optional[str] = None,
    annualize: bool = False,
    horizon_days: int = 5,
    periods_per_year: int = 252,
) -> pd.DataFrame:
    """
    저장된 예측값으로부터 BL용 Q wide matrix 생성

    기본적으로 y_pred를 그대로 사용.
    필요시 annualize 가능.
    """
    q = build_prediction_matrix(
        per_prediction=per_prediction,
        model_name=model_name,
        value_col="y_pred",
        fold_id=fold_id,
        evaluation_split=evaluation_split,
        dropna=False,
    )

    if annualize:
        q = (1.0 + q) ** (periods_per_year / horizon_days) - 1.0

    return q


def build_uncertainty_matrix_from_predictions(
    per_prediction: pd.DataFrame,
    model_name: str,
    fold_id: Optional[int] = None,
    evaluation_split: Optional[str] = None,
    square: bool = False,
) -> pd.DataFrame:
    """
    저장된 uncertainty로부터 wide matrix 생성
    square=True면 variance proxy로 사용 가능
    """
    omega = build_prediction_matrix(
        per_prediction=per_prediction,
        model_name=model_name,
        value_col="uncertainty",
        fold_id=fold_id,
        evaluation_split=evaluation_split,
        dropna=False,
    )

    if square:
        omega = omega ** 2

    return omega