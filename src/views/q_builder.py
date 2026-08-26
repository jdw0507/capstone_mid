from __future__ import annotations

from typing import Optional, Literal

import numpy as np
import pandas as pd


PairingMode = Literal["zip", "cartesian"]


class PredictionSnapshotEmptyError(ValueError):
    """Raised when no prediction snapshot exists for a requested key."""


def annualize_simple_return_forecast(
    x: pd.Series | pd.DataFrame | float,
    horizon_days: int = 5,
    periods_per_year: int = 252,
):
    """
    horizon 수익률 예측값을 연율화

    작은 수익률이면 선형근사도 가능하지만,
    여기서는 일관성을 위해 복리식 사용
    """
    return (1.0 + x) ** (periods_per_year / horizon_days) - 1.0


def get_prediction_snapshot(
    per_prediction: pd.DataFrame,
    model_name: str,
    date: pd.Timestamp | str,
    fold_id: Optional[int] = None,
    evaluation_split: Optional[str] = None,
    pred_col: str = "y_pred",
    uncertainty_col: str = "uncertainty",
) -> pd.DataFrame:
    """
    특정 모델/날짜의 asset별 예측값 snapshot 반환

    Returns
    -------
    pd.DataFrame
        columns:
        - asset
        - y_pred
        - uncertainty (있으면)
        - y_true / error (있으면 같이 유지)
    """
    if "model" not in per_prediction.columns or "asset" not in per_prediction.columns or "date" not in per_prediction.columns:
        raise ValueError("per_prediction에는 최소 model, asset, date 컬럼이 있어야 합니다.")

    df = per_prediction.copy()
    df["date"] = pd.to_datetime(df["date"])
    date = pd.Timestamp(date)

    df = df[df["model"] == model_name].copy()
    df = df[df["date"] == date].copy()

    if fold_id is not None:
        if "fold_id" not in df.columns:
            raise ValueError("fold_id 컬럼이 없습니다.")
        df = df[df["fold_id"] == fold_id].copy()

    if evaluation_split is not None:
        if "evaluation_split" not in df.columns:
            raise ValueError("evaluation_split 컬럼이 없습니다.")
        df = df[df["evaluation_split"] == evaluation_split].copy()

    keep_cols = ["asset", pred_col]
    for c in [uncertainty_col, "y_true", "error", "fold_id", "evaluation_split"]:
        if c in df.columns and c not in keep_cols:
            keep_cols.append(c)

    df = df[keep_cols].copy()
    df = df.rename(columns={pred_col: "q_value"})
    if uncertainty_col in df.columns:
        df = df.rename(columns={uncertainty_col: "uncertainty"})

    df = df.dropna(subset=["q_value"])
    df = df.sort_values("asset").reset_index(drop=True)

    if df.empty:
        raise PredictionSnapshotEmptyError(
            f"snapshot이 비어 있습니다. model={model_name}, date={date}, fold_id={fold_id}, split={evaluation_split}"
        )

    return df


def build_absolute_q(
    snapshot: pd.DataFrame,
    top_k: Optional[int] = None,
    bottom_k: int = 0,
    min_abs_pred: Optional[float] = None,
    annualize: bool = False,
    horizon_days: int = 5,
    periods_per_year: int = 252,
    allow_negative_absolute_views: bool = True,
) -> tuple[pd.Series, pd.DataFrame]:
    """
    absolute views용 Q 생성

    전략
    ----
    - top_k: 가장 높은 예측값 자산 선택
    - bottom_k: 가장 낮은 예측값 자산 선택
    - min_abs_pred: |예측값| 기준 필터
    - allow_negative_absolute_views=False면 음수 예측 자산은 제거

    Returns
    -------
    Q : pd.Series
        view_id index
    meta : pd.DataFrame
        view 메타정보
    """
    df = snapshot.copy()

    if min_abs_pred is not None:
        df = df[df["q_value"].abs() >= min_abs_pred].copy()

    top_df = df.sort_values("q_value", ascending=False)
    if top_k is not None:
        top_df = top_df.head(top_k)

    bottom_df = pd.DataFrame(columns=df.columns)
    if bottom_k > 0:
        bottom_df = df.sort_values("q_value", ascending=True).head(bottom_k)

    frames = []
    if not top_df.empty:
        frames.append(top_df)
    if not bottom_df.empty:
        frames.append(bottom_df)

    if frames:
        selected = pd.concat(frames, axis=0).drop_duplicates(subset=["asset"])
    else:
        selected = df.iloc[0:0].copy()
    selected = selected.sort_values("q_value", ascending=False).reset_index(drop=True)

    if not allow_negative_absolute_views:
        selected = selected[selected["q_value"] > 0].copy()

    if annualize:
        selected["q_value"] = annualize_simple_return_forecast(
            selected["q_value"],
            horizon_days=horizon_days,
            periods_per_year=periods_per_year,
        )

    if selected.empty:
        raise ValueError("absolute Q를 만들 수 있는 자산이 없습니다.")

    selected["view_id"] = [f"abs_{i}" for i in range(len(selected))]
    selected["view_type"] = "absolute"

    Q = pd.Series(selected["q_value"].values, index=selected["view_id"], name="Q")
    meta = selected[["view_id", "view_type", "asset", "q_value"]].copy()

    if "uncertainty" in selected.columns:
        meta["uncertainty"] = selected["uncertainty"].values
    if "error" in selected.columns:
        meta["error"] = selected["error"].values

    return Q, meta


def build_relative_q(
    snapshot: pd.DataFrame,
    top_k: int = 3,
    bottom_k: int = 3,
    pairing: PairingMode = "zip",
    annualize: bool = False,
    horizon_days: int = 5,
    periods_per_year: int = 252,
) -> tuple[pd.Series, pd.DataFrame]:
    """
    relative views용 Q 생성

    예:
    top 자산이 bottom 자산보다 얼마나 더 높은 expected return인가

    pairing
    -------
    - "zip": top 순위와 bottom 순위를 1:1 대응
    - "cartesian": top x bottom 모든 조합
    """
    df = snapshot.copy().sort_values("q_value", ascending=False)

    top_df = df.head(top_k).copy()
    bottom_df = df.tail(bottom_k).sort_values("q_value", ascending=True).copy()

    if top_df.empty or bottom_df.empty:
        raise ValueError("relative Q를 만들기 위한 top/bottom 자산이 부족합니다.")

    rows = []

    if pairing == "zip":
        n = min(len(top_df), len(bottom_df))
        for i in range(n):
            long_asset = top_df.iloc[i]
            short_asset = bottom_df.iloc[i]

            q_val = float(long_asset["q_value"]) - float(short_asset["q_value"])
            rows.append(
                {
                    "view_id": f"rel_{i}",
                    "view_type": "relative",
                    "long_asset": long_asset["asset"],
                    "short_asset": short_asset["asset"],
                    "q_value": q_val,
                    "long_uncertainty": long_asset["uncertainty"] if "uncertainty" in top_df.columns else np.nan,
                    "short_uncertainty": short_asset["uncertainty"] if "uncertainty" in bottom_df.columns else np.nan,
                }
            )

    elif pairing == "cartesian":
        idx = 0
        for _, long_asset in top_df.iterrows():
            for _, short_asset in bottom_df.iterrows():
                q_val = float(long_asset["q_value"]) - float(short_asset["q_value"])
                rows.append(
                    {
                        "view_id": f"rel_{idx}",
                        "view_type": "relative",
                        "long_asset": long_asset["asset"],
                        "short_asset": short_asset["asset"],
                        "q_value": q_val,
                        "long_uncertainty": long_asset["uncertainty"] if "uncertainty" in top_df.columns else np.nan,
                        "short_uncertainty": short_asset["uncertainty"] if "uncertainty" in bottom_df.columns else np.nan,
                    }
                )
                idx += 1
    else:
        raise ValueError(f"지원하지 않는 pairing: {pairing}")

    rel_df = pd.DataFrame(rows)

    if annualize:
        rel_df["q_value"] = annualize_simple_return_forecast(
            rel_df["q_value"],
            horizon_days=horizon_days,
            periods_per_year=periods_per_year,
        )

    Q = pd.Series(rel_df["q_value"].values, index=rel_df["view_id"], name="Q")
    meta = rel_df.copy()

    return Q, meta


def build_mixed_q(
    snapshot: pd.DataFrame,
    absolute_top_k: int = 3,
    absolute_bottom_k: int = 0,
    relative_top_k: int = 2,
    relative_bottom_k: int = 2,
    relative_pairing: PairingMode = "zip",
    annualize: bool = False,
    horizon_days: int = 5,
    periods_per_year: int = 252,
) -> tuple[pd.Series, pd.DataFrame]:
    """
    absolute + relative view를 함께 만드는 convenience 함수
    """
    q_abs, meta_abs = build_absolute_q(
        snapshot=snapshot,
        top_k=absolute_top_k,
        bottom_k=absolute_bottom_k,
        annualize=annualize,
        horizon_days=horizon_days,
        periods_per_year=periods_per_year,
    )

    q_rel, meta_rel = build_relative_q(
        snapshot=snapshot,
        top_k=relative_top_k,
        bottom_k=relative_bottom_k,
        pairing=relative_pairing,
        annualize=annualize,
        horizon_days=horizon_days,
        periods_per_year=periods_per_year,
    )

    Q = pd.concat([q_abs, q_rel], axis=0)
    meta = pd.concat([meta_abs, meta_rel], axis=0, ignore_index=True)

    return Q, meta
