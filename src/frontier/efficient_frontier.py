from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import minimize


def portfolio_return(weights: pd.Series, expected_returns: pd.Series) -> float:
    return float(weights @ expected_returns)


def portfolio_variance(weights: pd.Series, covariance: pd.DataFrame) -> float:
    w = weights.values
    cov = covariance.loc[weights.index, weights.index].values
    return float(w.T @ cov @ w)


def portfolio_volatility(weights: pd.Series, covariance: pd.DataFrame) -> float:
    return float(np.sqrt(portfolio_variance(weights, covariance)))


def solve_min_variance_for_target_return(
    expected_returns: pd.Series,
    covariance: pd.DataFrame,
    target_return: float,
    long_only: bool = True,
    weight_bounds: Optional[Tuple[float, float]] = None,
) -> pd.Series:
    """
    target_return을 만족하는 최소분산 포트폴리오를 계산.
    """

    _validate_inputs(expected_returns, covariance)

    assets = expected_returns.index.tolist()
    mu = expected_returns.loc[assets].values
    cov = covariance.loc[assets, assets].values
    n_assets = len(assets)

    x0 = np.ones(n_assets) / n_assets
    bounds = _build_bounds(n_assets, long_only=long_only, weight_bounds=weight_bounds)

    constraints = [
        {"type": "eq", "fun": lambda w: np.sum(w) - 1.0},
        {"type": "eq", "fun": lambda w: float(w @ mu) - target_return},
    ]

    result = minimize(
        fun=lambda w: float(w.T @ cov @ w),
        x0=x0,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
    )

    if not result.success:
        raise ValueError(f"Target return 최적화 실패: {result.message}")

    weights = pd.Series(result.x, index=assets, name="weight")
    weights = weights / weights.sum()
    return weights


def build_efficient_frontier(
    expected_returns: pd.Series,
    covariance: pd.DataFrame,
    n_points: int = 50,
    long_only: bool = True,
    weight_bounds: Optional[Tuple[float, float]] = None,
) -> pd.DataFrame:
    """
    여러 target return 구간에 대해 efficient frontier 구성.

    Returns
    -------
    pd.DataFrame
        columns:
        - target_return
        - volatility
        - variance
        - sharpe_zero_rf
    """
    _validate_inputs(expected_returns, covariance)

    mu = expected_returns.sort_index()
    cov = covariance.loc[mu.index, mu.index]

    min_ret = float(mu.min())
    max_ret = float(mu.max())

    target_returns = np.linspace(min_ret, max_ret, n_points)

    rows = []
    for target in target_returns:
        try:
            weights = solve_min_variance_for_target_return(
                expected_returns=mu,
                covariance=cov,
                target_return=float(target),
                long_only=long_only,
                weight_bounds=weight_bounds,
            )

            port_ret = portfolio_return(weights, mu)
            port_var = portfolio_variance(weights, cov)
            port_vol = np.sqrt(port_var)
            sharpe_zero_rf = port_ret / port_vol if port_vol > 0 else np.nan

            row = {
                "target_return": port_ret,
                "volatility": port_vol,
                "variance": port_var,
                "sharpe_zero_rf": sharpe_zero_rf,
            }

            for asset, w in weights.items():
                row[f"weight_{asset}"] = w

            rows.append(row)

        except Exception:
            continue

    frontier = pd.DataFrame(rows)
    if frontier.empty:
        raise ValueError("Efficient frontier를 생성하지 못했습니다.")

    frontier = frontier.sort_values("volatility").reset_index(drop=True)
    return frontier


def _build_bounds(
    n_assets: int,
    long_only: bool = True,
    weight_bounds: Optional[Tuple[float, float]] = None,
):
    if weight_bounds is not None:
        return [weight_bounds] * n_assets

    if long_only:
        return [(0.0, 1.0)] * n_assets

    return [(-1.0, 1.0)] * n_assets


def _validate_inputs(
    expected_returns: pd.Series,
    covariance: pd.DataFrame,
) -> None:
    if not isinstance(expected_returns, pd.Series):
        raise TypeError("expected_returns는 pandas Series여야 합니다.")

    if not isinstance(covariance, pd.DataFrame):
        raise TypeError("covariance는 pandas DataFrame이어야 합니다.")

    if expected_returns.empty:
        raise ValueError("expected_returns가 비어 있습니다.")

    if covariance.empty:
        raise ValueError("covariance가 비어 있습니다.")

    if expected_returns.isna().any():
        raise ValueError("expected_returns에 NaN이 있습니다.")

    if covariance.isna().any().any():
        raise ValueError("covariance에 NaN이 있습니다.")

    if set(expected_returns.index) != set(covariance.index) or set(covariance.index) != set(covariance.columns):
        raise ValueError("expected_returns와 covariance의 자산 구성이 일치해야 합니다.")
    



def generate_random_portfolios(
    expected_returns: pd.Series,
    covariance: pd.DataFrame,
    n_samples: int = 5000,
    long_only: bool = True,
    weight_bounds: tuple[float, float] | None = None,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    랜덤 포트폴리오 샘플 생성.
    hover용으로 각 자산 비중까지 함께 저장.

    Returns
    -------
    pd.DataFrame
        columns:
        - expected_return
        - volatility
        - variance
        - sharpe_zero_rf
        - weight_{asset}
    """
    _validate_inputs(expected_returns, covariance)

    rng = np.random.default_rng(random_state)

    mu = expected_returns.sort_index()
    cov = covariance.loc[mu.index, mu.index]
    assets = mu.index.tolist()
    n_assets = len(assets)

    rows = []

    for _ in range(n_samples):
        weights = _sample_weights(
            n_assets=n_assets,
            rng=rng,
            long_only=long_only,
            weight_bounds=weight_bounds,
        )

        w = pd.Series(weights, index=assets, name="weight")
        port_ret = portfolio_return(w, mu)
        port_var = portfolio_variance(w, cov)
        port_vol = np.sqrt(port_var)
        sharpe_zero_rf = port_ret / port_vol if port_vol > 0 else np.nan

        row = {
            "expected_return": port_ret,
            "volatility": port_vol,
            "variance": port_var,
            "sharpe_zero_rf": sharpe_zero_rf,
        }

        for asset, val in w.items():
            row[f"weight_{asset}"] = val

        rows.append(row)

    return pd.DataFrame(rows)


def _sample_weights(
    n_assets: int,
    rng: np.random.Generator,
    long_only: bool = True,
    weight_bounds: tuple[float, float] | None = None,
) -> np.ndarray:
    """
    랜덤 weight 생성.
    long_only + bounds가 있으면 rejection sampling으로 처리.
    """
    if long_only:
        if weight_bounds is None:
            w = rng.random(n_assets)
            w = w / w.sum()
            return w

        lower, upper = weight_bounds

        for _ in range(10000):
            w = rng.random(n_assets)
            w = w / w.sum()
            if np.all(w >= lower) and np.all(w <= upper):
                return w

        raise ValueError("주어진 weight_bounds로 랜덤 포트폴리오 생성에 실패했습니다.")

    # short 허용이면 일단 정규분포 기반 후 정규화
    w = rng.normal(size=n_assets)
    if np.isclose(w.sum(), 0.0):
        w[0] += 1e-6
    w = w / w.sum()

    if weight_bounds is not None:
        lower, upper = weight_bounds
        for _ in range(10000):
            w = rng.normal(size=n_assets)
            if np.isclose(w.sum(), 0.0):
                w[0] += 1e-6
            w = w / w.sum()
            if np.all(w >= lower) and np.all(w <= upper):
                return w
        raise ValueError("주어진 weight_bounds로 랜덤 포트폴리오 생성에 실패했습니다.")

    return w