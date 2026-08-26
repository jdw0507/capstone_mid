from __future__ import annotations

from functools import lru_cache
from typing import Optional
import warnings

import pandas as pd
from pandas_datareader.data import DataReader
from pandas_datareader.famafrench import get_available_datasets


def _to_month_end_index(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    if isinstance(out.index, pd.PeriodIndex):
        out.index = out.index.to_timestamp(how="end").normalize()
    else:
        out.index = pd.to_datetime(out.index)
        out.index = out.index.to_period("M").to_timestamp("M")

    out = out.sort_index()
    out = out[~out.index.duplicated(keep="first")]
    return out


def _standardize_factor_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    new_cols = []
    for col in out.columns:
        c = str(col).strip()

        if c in {"Mkt-RF", "Mkt_RF", "MKT-RF", "MKT_RF"}:
            new_cols.append("MKT_RF")
        elif c.upper() == "RF":
            new_cols.append("RF")
        elif c.upper() == "SMB":
            new_cols.append("SMB")
        elif c.upper() == "HML":
            new_cols.append("HML")
        elif c.upper() in {"MOM", "UMD"}:
            new_cols.append("MOM")
        else:
            new_cols.append(c.upper().replace("-", "_").replace(" ", "_"))

    out.columns = new_cols
    return out


@lru_cache(maxsize=32)
def _load_monthly_famafrench_dataset_cached(
    dataset_name: str,
    start: Optional[str] = "1900-01-01",
    end: Optional[str] = None,
) -> pd.DataFrame:
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=".*date_parser.*deprecated.*",
            category=FutureWarning,
        )
        raw = DataReader(dataset_name, "famafrench", start=start, end=end)

    df = raw[0].copy()
    df = _to_month_end_index(df)
    df = _standardize_factor_columns(df)
    df = df.apply(pd.to_numeric, errors="coerce")
    df = df / 100.0
    df = df.dropna(how="all")

    return df


def _load_monthly_famafrench_dataset(
    dataset_name: str,
    start: Optional[str] = "1900-01-01",
    end: Optional[str] = None,
) -> pd.DataFrame:
    return _load_monthly_famafrench_dataset_cached(dataset_name, start, end).copy()


@lru_cache(maxsize=1)
def _find_monthly_momentum_dataset_name() -> str:
    datasets = get_available_datasets()

    preferred = [ds for ds in datasets if ds.lower() == "f-f_momentum_factor"]
    if preferred:
        return preferred[0]

    candidates = [
        ds for ds in datasets
        if "momentum_factor" in ds.lower() and "daily" not in ds.lower()
    ]
    if candidates:
        return candidates[0]

    fallback = [
        ds for ds in datasets
        if "mom" in ds.lower() and "factor" in ds.lower() and "daily" not in ds.lower()
    ]
    if fallback:
        return fallback[0]

    raise ValueError("월별 Momentum factor dataset 이름을 찾지 못했습니다.")


def load_capm_factors(
    start: Optional[str] = "1900-01-01",
    end: Optional[str] = None,
) -> pd.DataFrame:
    ff3 = _load_monthly_famafrench_dataset(
        dataset_name="F-F_Research_Data_Factors",
        start=start,
        end=end,
    )
    return ff3[["MKT_RF", "RF"]].copy()


def load_ff3_factors(
    start: Optional[str] = "1900-01-01",
    end: Optional[str] = None,
) -> pd.DataFrame:
    ff3 = _load_monthly_famafrench_dataset(
        dataset_name="F-F_Research_Data_Factors",
        start=start,
        end=end,
    )
    return ff3[["MKT_RF", "SMB", "HML", "RF"]].copy()


def load_carhart_factors(
    start: Optional[str] = "1900-01-01",
    end: Optional[str] = None,
) -> pd.DataFrame:
    ff3 = load_ff3_factors(start=start, end=end)

    mom_dataset = _find_monthly_momentum_dataset_name()
    mom = _load_monthly_famafrench_dataset(
        dataset_name=mom_dataset,
        start=start,
        end=end,
    )

    if "MOM" not in mom.columns:
        mom_candidates = [c for c in mom.columns if "MOM" in c or "UMD" in c]
        if not mom_candidates:
            raise ValueError("Momentum factor column을 찾지 못했습니다.")
        mom = mom.rename(columns={mom_candidates[0]: "MOM"})

    factors = ff3.join(mom[["MOM"]], how="inner")
    return factors[["MKT_RF", "SMB", "HML", "MOM", "RF"]].copy()