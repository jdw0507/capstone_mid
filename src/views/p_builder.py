from __future__ import annotations

import pandas as pd


def build_absolute_p(
    assets: list[str],
    meta_abs: pd.DataFrame,
) -> pd.DataFrame:
    """
    absolute view용 P 생성

    meta_abs 필수 컬럼:
    - view_id
    - asset
    """
    required = {"view_id", "asset"}
    missing = required - set(meta_abs.columns)
    if missing:
        raise ValueError(f"meta_abs에 필요한 컬럼이 없습니다: {missing}")

    assets = list(assets)
    P = pd.DataFrame(0.0, index=meta_abs["view_id"].tolist(), columns=assets)

    for _, row in meta_abs.iterrows():
        asset = row["asset"]
        if asset not in P.columns:
            continue
        P.loc[row["view_id"], asset] = 1.0

    return P


def build_relative_p(
    assets: list[str],
    meta_rel: pd.DataFrame,
) -> pd.DataFrame:
    """
    relative view용 P 생성

    meta_rel 필수 컬럼:
    - view_id
    - long_asset
    - short_asset
    """
    required = {"view_id", "long_asset", "short_asset"}
    missing = required - set(meta_rel.columns)
    if missing:
        raise ValueError(f"meta_rel에 필요한 컬럼이 없습니다: {missing}")

    assets = list(assets)
    P = pd.DataFrame(0.0, index=meta_rel["view_id"].tolist(), columns=assets)

    for _, row in meta_rel.iterrows():
        long_asset = row["long_asset"]
        short_asset = row["short_asset"]

        if long_asset in P.columns:
            P.loc[row["view_id"], long_asset] = 1.0
        if short_asset in P.columns:
            P.loc[row["view_id"], short_asset] = -1.0

    return P


def combine_p_matrices(*p_list: pd.DataFrame) -> pd.DataFrame:
    """
    여러 P 행렬을 row 기준으로 결합
    """
    valid = [p for p in p_list if p is not None and not p.empty]
    if len(valid) == 0:
        return pd.DataFrame()
    return pd.concat(valid, axis=0)