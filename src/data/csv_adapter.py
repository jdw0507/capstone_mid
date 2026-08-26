from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import pandas as pd


@dataclass
class CSVSchema:
    date_col: str
    assets: list[str]
    shared_columns: list[str]
    asset_columns: list[str]
    asset_feature_map: dict[str, list[str]]
    target_prefix: str


def load_final_clean_result(
    path: str | Path,
    date_col: str = "Date",
    sort: bool = True,
    drop_duplicate_dates: bool = True,
) -> pd.DataFrame:
    """
    final_clean_result.csv 로드

    Parameters
    ----------
    path : str | Path
        CSV 파일 경로
    date_col : str
        날짜 컬럼명
    sort : bool
        날짜 정렬 여부
    drop_duplicate_dates : bool
        중복 날짜 제거 여부

    Returns
    -------
    pd.DataFrame
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"파일을 찾을 수 없습니다: {path}")

    df = pd.read_csv(path)

    if date_col not in df.columns:
        raise ValueError(f"date_col={date_col} 컬럼이 없습니다.")

    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.dropna(subset=[date_col])

    if drop_duplicate_dates:
        df = df.drop_duplicates(subset=[date_col], keep="first")

    if sort:
        df = df.sort_values(date_col).reset_index(drop=True)

    return df


def infer_asset_universe(
    df: pd.DataFrame,
    close_prefix: str = "Close_",
    target_prefix: str = "Return_5d_",
) -> list[str]:
    """
    wide CSV 컬럼명에서 자산 유니버스 추론

    우선순위:
    1. Close_<ticker>
    2. Return_5d_<ticker>

    Returns
    -------
    list[str]
    """
    assets = []

    close_assets = [c.replace(close_prefix, "") for c in df.columns if c.startswith(close_prefix)]
    target_assets = [c.replace(target_prefix, "") for c in df.columns if c.startswith(target_prefix)]

    merged = close_assets + target_assets
    seen = set()
    for a in merged:
        a = str(a).strip()
        if a and a not in seen:
            assets.append(a)
            seen.add(a)

    if len(assets) == 0:
        raise ValueError("Close_ 또는 Return_5d_ 패턴에서 자산 유니버스를 추론하지 못했습니다.")

    return assets


def is_asset_specific_column(
    col: str,
    assets: Iterable[str],
    date_col: str = "Date",
) -> bool:
    """
    컬럼이 특정 자산에 속하는 컬럼인지 판별
    예: SMA_20_AAPL, Return_5d_MSFT
    """
    if col == date_col:
        return False

    for asset in sorted(set(assets), key=len, reverse=True):
        if col.endswith(f"_{asset}"):
            return True
    return False


def strip_asset_suffix(
    col: str,
    assets: Iterable[str],
) -> str:
    """
    예:
    SMA_20_AAPL -> SMA_20
    Return_5d_MSFT -> Return_5d
    Close_NVDA -> Close
    """
    for asset in sorted(set(assets), key=len, reverse=True):
        suffix = f"_{asset}"
        if col.endswith(suffix):
            return col[: -len(suffix)]
    return col


def build_csv_schema(
    df: pd.DataFrame,
    date_col: str = "Date",
    target_prefix: str = "Return_5d_",
) -> CSVSchema:
    """
    CSV 전체 구조를 분석해서 schema 반환
    """
    assets = infer_asset_universe(df)
    asset_columns = [c for c in df.columns if is_asset_specific_column(c, assets, date_col=date_col)]
    shared_columns = [c for c in df.columns if c not in asset_columns and c != date_col]

    asset_feature_map: dict[str, list[str]] = {}
    for asset in assets:
        asset_cols = [c for c in asset_columns if c.endswith(f"_{asset}")]
        asset_feature_map[asset] = sorted(asset_cols)

    return CSVSchema(
        date_col=date_col,
        assets=assets,
        shared_columns=shared_columns,
        asset_columns=asset_columns,
        asset_feature_map=asset_feature_map,
        target_prefix=target_prefix,
    )


def build_asset_panel(
    df: pd.DataFrame,
    assets: Optional[Iterable[str]] = None,
    date_col: str = "Date",
    target_prefix: str = "Return_5d_",
    target_name: str = "target_5d",
    include_shared: bool = True,
    include_close: bool = True,
    dropna_target: bool = True,
    drop_all_feature_nan_rows: bool = False,
) -> pd.DataFrame:
    """
    wide CSV를 asset-panel(long format)로 변환

    Output columns 예시
    -------------------
    date, asset, close, target_5d, SMA_20, EMA_20, ..., Shanghai, Brent_Oil, ...

    Parameters
    ----------
    df : pd.DataFrame
        원본 wide format 데이터
    assets : list[str] | None
        사용할 자산 subset. None이면 전체 자산 사용
    date_col : str
        날짜 컬럼명
    target_prefix : str
        타깃 prefix (기본: Return_5d_)
    target_name : str
        패널에서 사용할 타깃 컬럼명
    include_shared : bool
        공통 설명변수 포함 여부
    include_close : bool
        close 컬럼 포함 여부
    dropna_target : bool
        타깃이 없는 행 제거 여부
    drop_all_feature_nan_rows : bool
        설명변수가 전부 NaN인 행 제거 여부
    """
    if date_col not in df.columns:
        raise ValueError(f"date_col={date_col} 컬럼이 없습니다.")

    schema = build_csv_schema(df, date_col=date_col, target_prefix=target_prefix)

    if assets is None:
        assets = schema.assets
    else:
        assets = [a for a in assets if a in schema.assets]

    if len(assets) == 0:
        raise ValueError("사용 가능한 자산이 없습니다.")

    shared_cols = schema.shared_columns if include_shared else []

    panel_parts = []
    for asset in assets:
        asset_cols = schema.asset_feature_map[asset]
        use_cols = [date_col] + asset_cols + shared_cols

        part = df[use_cols].copy()
        part["asset"] = asset

        rename_map = {}
        for col in asset_cols:
            base_name = strip_asset_suffix(col, schema.assets)

            if col == f"{target_prefix}{asset}":
                rename_map[col] = target_name
            elif base_name == "Close":
                rename_map[col] = "close"
            else:
                rename_map[col] = base_name

        part = part.rename(columns=rename_map)
        part = part.rename(columns={date_col: "date"})

        if not include_close and "close" in part.columns:
            part = part.drop(columns=["close"])

        if dropna_target and target_name in part.columns:
            part = part.dropna(subset=[target_name])

        non_meta_cols = [c for c in part.columns if c not in {"date", "asset", target_name}]
        if drop_all_feature_nan_rows and len(non_meta_cols) > 0:
            part = part.dropna(subset=non_meta_cols, how="all")

        panel_parts.append(part)

    panel = pd.concat(panel_parts, axis=0, ignore_index=True)
    panel["date"] = pd.to_datetime(panel["date"])
    panel = panel.sort_values(["date", "asset"]).reset_index(drop=True)

    return panel


def get_feature_columns(
    panel_df: pd.DataFrame,
    target_col: str = "target_5d",
    exclude_cols: Optional[Iterable[str]] = None,
    exclude_target_like: bool = True,
) -> list[str]:
    """
    패널 데이터에서 feature 컬럼명만 추출

    exclude_target_like=True이면
    target / future return 관련 컬럼을 전부 제거한다.
    """
    default_exclude = {"date", "asset", target_col}

    if exclude_cols is not None:
        default_exclude.update(exclude_cols)

    feature_cols = [c for c in panel_df.columns if c not in default_exclude]

    if exclude_target_like:
        filtered = []
        for c in feature_cols:
            c_lower = c.lower()

            is_target_like = (
                c_lower.startswith("target")
                or c_lower.startswith("return_")
                or c_lower.startswith("future_")
                or c_lower.endswith("_target")
                or "excess" in c_lower and c_lower.startswith("target")
            )

            if not is_target_like:
                filtered.append(c)

        feature_cols = filtered

    return feature_cols


def make_model_matrices(
    panel_df: pd.DataFrame,
    target_col: str = "target_5d",
    feature_cols: Optional[Iterable[str]] = None,
    exclude_target_like: bool = True,
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """
    패널 데이터 -> X, y, meta 분리
    """
    if target_col not in panel_df.columns:
        raise ValueError(f"{target_col} 컬럼이 없습니다.")

    if feature_cols is None:
        feature_cols = get_feature_columns(
            panel_df,
            target_col=target_col,
            exclude_target_like=exclude_target_like,
        )

    feature_cols = list(feature_cols)

    X = panel_df[feature_cols].copy()
    y = panel_df[target_col].copy()
    meta = panel_df[["date", "asset"]].copy()

    return X, y, meta


def make_wide_target_table(
    panel_df: pd.DataFrame,
    target_col: str = "target_5d",
) -> pd.DataFrame:
    """
    panel -> date x asset 타깃 테이블
    """
    if target_col not in panel_df.columns:
        raise ValueError(f"{target_col} 컬럼이 없습니다.")

    out = panel_df.pivot(index="date", columns="asset", values=target_col)
    out = out.sort_index().sort_index(axis=1)
    return out