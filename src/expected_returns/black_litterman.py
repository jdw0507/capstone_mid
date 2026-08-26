from __future__ import annotations

from typing import Optional, Sequence, Literal

import numpy as np
import pandas as pd
import yfinance as yf

from .base import ExpectedReturnModel


FallbackType = Literal["equal_weight", "inverse_vol"]


def fetch_market_caps_from_yahoo(
    tickers: Sequence[str],
) -> pd.Series:
    """
    Fetch market caps from Yahoo Finance for each ticker.
    """
    caps = {}
    for t in tickers:
        caps[t] = _fetch_single_market_cap(str(t).strip().upper())
    return pd.Series(caps, name="market_cap").astype(float)


def _fetch_single_market_cap(ticker: str) -> float | None:
    """
    Market cap priority:
    1) fast_info.market_cap
    2) info["marketCap"]
    3) shares_full * latest_price
    """
    try:
        tk = yf.Ticker(ticker)

        # fast_info
        try:
            fi = tk.fast_info
            if fi is not None:
                if isinstance(fi, dict):
                    mc = fi.get("market_cap", None)
                else:
                    mc = getattr(fi, "market_cap", None)
                if mc is not None and pd.notna(mc):
                    return float(mc)
        except Exception:
            pass

        # info
        try:
            info = tk.info
            mc = info.get("marketCap", None)
            if mc is not None and pd.notna(mc):
                return float(mc)
        except Exception:
            pass

        # shares * latest price
        shares = None
        latest_price = None

        try:
            shares_df = tk.get_shares_full(start="1900-01-01")
            if shares_df is not None and len(shares_df) > 0:
                shares = float(shares_df.dropna().iloc[-1])
        except Exception:
            shares = None

        try:
            hist = tk.history(period="7d", auto_adjust=False)
            if hist is not None and not hist.empty:
                col = "Adj Close" if "Adj Close" in hist.columns else "Close"
                latest_price = float(hist[col].dropna().iloc[-1])
        except Exception:
            latest_price = None

        if shares is not None and latest_price is not None:
            return float(shares * latest_price)

    except Exception:
        pass

    return None


def build_universe_market_weights(
    assets: list[str],
    market_weights: Optional[pd.Series] = None,
    market_caps: Optional[pd.Series] = None,
    covariance: Optional[pd.DataFrame] = None,
    fallback: FallbackType = "equal_weight",
    blend_with_equal_weight: float | None = None,
) -> pd.Series:
    """
    Build market weights normalized within the investable universe.

    Priority:
    1) explicit market_weights
    2) market_caps normalized within the universe
    3) fallback
    """
    assets = list(assets)

    if market_weights is not None:
        w = market_weights.reindex(assets).astype(float).fillna(0.0)
        if np.isclose(w.sum(), 0.0):
            raise ValueError("market_weights sum is zero.")
        w = w / w.sum()

    elif market_caps is not None:
        caps = market_caps.reindex(assets).astype(float).fillna(0.0)
        if np.isclose(caps.sum(), 0.0):
            w = _fallback_weights(assets, covariance=covariance, fallback=fallback)
        else:
            w = caps / caps.sum()

    else:
        w = _fallback_weights(assets, covariance=covariance, fallback=fallback)

    if blend_with_equal_weight is not None:
        lam = float(blend_with_equal_weight)
        if not (0.0 <= lam <= 1.0):
            raise ValueError("blend_with_equal_weight must be between 0 and 1.")
        w_eq = pd.Series(1.0 / len(assets), index=assets)
        w = lam * w + (1.0 - lam) * w_eq
        w = w / w.sum()

    w.name = "market_weight"
    return w


def _fallback_weights(
    assets: list[str],
    covariance: Optional[pd.DataFrame] = None,
    fallback: FallbackType = "equal_weight",
) -> pd.Series:
    if fallback == "equal_weight":
        return pd.Series(1.0 / len(assets), index=assets, name="market_weight")

    if fallback == "inverse_vol":
        if covariance is None:
            raise ValueError("covariance is required for inverse_vol fallback.")
        cov = covariance.loc[assets, assets]
        vol = np.sqrt(np.diag(cov.values))
        vol = pd.Series(vol, index=assets).replace(0.0, np.nan).fillna(vol.mean())
        inv_vol = 1.0 / vol
        w = inv_vol / inv_vol.sum()
        w.name = "market_weight"
        return w

    raise ValueError(f"Unsupported fallback: {fallback}")


class BlackLittermanExpectedReturn(ExpectedReturnModel):
    """
    Black-Litterman expected return model with explicit market weight support.
    """

    def __init__(
        self,
        tickers: Sequence[str],
        prices: Optional[pd.DataFrame] = None,
        periods_per_year: int = 252,
        min_obs: int = 252,
        tau: float = 0.05,
        risk_free_rate: float = 0.02,
        market_ticker: str = "SPY",
        market_prices: Optional[pd.Series] = None,
        market_caps: Optional[pd.Series] = None,
        market_weights: Optional[pd.Series] = None,
        risk_aversion: Optional[float] = 2.5,
        P: Optional[object] = None,
        Q: Optional[object] = None,
        Omega: Optional[object] = None,
        view_confidences: Optional[object] = None,
        use_market_cap_weights: bool = True,
        allow_equal_weight_fallback: bool = True,
        market_weight_fallback: FallbackType = "equal_weight",
        blend_with_equal_weight: float | None = None,
    ) -> None:
        self.tickers = [str(t).strip().upper() for t in tickers]
        self.prices = prices.copy() if prices is not None else None

        self.periods_per_year = periods_per_year
        self.min_obs = min_obs
        self.tau = tau
        self.risk_free_rate = risk_free_rate

        self.market_ticker = market_ticker.upper()
        self.market_prices = market_prices.copy() if market_prices is not None else None
        self.market_caps = market_caps.copy() if market_caps is not None else None
        self.market_weights = market_weights.copy() if market_weights is not None else None
        self.risk_aversion = risk_aversion

        self.P = P
        self.Q = Q
        self.Omega = Omega
        self.view_confidences = view_confidences

        self.use_market_cap_weights = use_market_cap_weights
        self.allow_equal_weight_fallback = allow_equal_weight_fallback
        self.market_weight_fallback = market_weight_fallback
        self.blend_with_equal_weight = blend_with_equal_weight

        # diagnostics
        self.last_assets_: list[str] | None = None
        self.last_covariance_: pd.DataFrame | None = None
        self.last_market_caps_: pd.Series | None = None
        self.last_market_weights_: pd.Series | None = None
        self.last_market_weights_source_: str | None = None
        self.last_risk_aversion_: float | None = None
        self.last_prior_: pd.Series | None = None
        self.last_posterior_: pd.Series | None = None
        self.last_omega_: pd.DataFrame | None = None
        self.last_used_equal_weight_fallback_: bool = False

    def fit_predict(self, returns: pd.DataFrame) -> pd.Series:
        returns = self._prepare_returns(returns)

        valid_counts = returns.notna().sum()
        eligible_cols = valid_counts[valid_counts >= self.min_obs].index.tolist()
        if len(eligible_cols) == 0:
            raise ValueError("No assets satisfy min_obs.")

        ret = returns[eligible_cols].copy().sort_index(axis=1)
        assets = ret.columns.tolist()

        cov = ret.cov() * self.periods_per_year
        cov = cov.loc[assets, assets]

        market_caps = self._resolve_market_caps(assets)
        market_weights = self._resolve_market_weights(assets, cov, market_caps)

        delta = self._get_risk_aversion(ret.index)

        prior = pd.Series(
            delta * (cov.values @ market_weights.values),
            index=assets,
            name="expected_return",
        )

        self.last_assets_ = assets
        self.last_covariance_ = cov.copy()
        self.last_market_caps_ = market_caps.copy() if market_caps is not None else None
        self.last_market_weights_ = market_weights.copy()
        self.last_risk_aversion_ = float(delta)
        self.last_prior_ = prior.copy()

        if self.P is None or self.Q is None:
            self.last_posterior_ = prior.copy()
            return prior

        posterior = self._compute_posterior(
            assets=assets,
            cov=cov,
            prior=prior,
        )

        self.last_posterior_ = posterior.copy()
        return posterior

    def _prepare_returns(self, returns: pd.DataFrame) -> pd.DataFrame:
        if not isinstance(returns, pd.DataFrame):
            raise TypeError("returns must be a pandas DataFrame.")
        if returns.empty:
            raise ValueError("returns is empty.")

        out = returns.copy()
        out.index = pd.to_datetime(out.index)
        out = out.sort_index()
        out = out[~out.index.duplicated(keep="first")]
        out.columns = [str(c).strip().upper() for c in out.columns]
        out = out.apply(pd.to_numeric, errors="coerce")

        common = [c for c in out.columns if c in self.tickers]
        if len(common) == 0:
            raise ValueError("No overlap between returns columns and tickers.")
        out = out[common]

        return out

    def _resolve_market_caps(self, assets: list[str]) -> pd.Series | None:
        if self.market_caps is not None:
            caps = self.market_caps.reindex(assets).astype(float)
            return caps.fillna(0.0)

        if not self.use_market_cap_weights:
            return None

        caps = fetch_market_caps_from_yahoo(assets)
        if caps.notna().sum() == 0:
            return None
        return caps.fillna(0.0)

    def _resolve_market_weights(
        self,
        assets: list[str],
        cov: pd.DataFrame,
        market_caps: pd.Series | None,
    ) -> pd.Series:
        if self.market_weights is not None:
            self.last_market_weights_source_ = "explicit_market_weights"
            w = build_universe_market_weights(
                assets=assets,
                market_weights=self.market_weights,
                market_caps=None,
                covariance=cov,
                fallback=self.market_weight_fallback,
                blend_with_equal_weight=self.blend_with_equal_weight,
            )
            return w

        if market_caps is not None and market_caps.sum() > 0:
            self.last_market_weights_source_ = "universe_market_caps"
            w = build_universe_market_weights(
                assets=assets,
                market_weights=None,
                market_caps=market_caps,
                covariance=cov,
                fallback=self.market_weight_fallback,
                blend_with_equal_weight=self.blend_with_equal_weight,
            )
            return w

        if self.allow_equal_weight_fallback:
            self.last_market_weights_source_ = f"fallback_{self.market_weight_fallback}"
            if self.market_weight_fallback == "equal_weight":
                self.last_used_equal_weight_fallback_ = True

            w = build_universe_market_weights(
                assets=assets,
                market_weights=None,
                market_caps=None,
                covariance=cov,
                fallback=self.market_weight_fallback,
                blend_with_equal_weight=self.blend_with_equal_weight,
            )
            return w

        raise ValueError("Failed to resolve market weights. Provide market_weights or market_caps.")

    def _get_risk_aversion(self, returns_index: pd.Index) -> float:
        if self.risk_aversion is not None:
            return float(self.risk_aversion)
        return 2.5

    def _compute_posterior(
        self,
        assets: list[str],
        cov: pd.DataFrame,
        prior: pd.Series,
    ) -> pd.Series:
        sigma = cov.values
        pi = prior.values.reshape(-1, 1)

        P = self._to_2d_array(self.P)
        Q = self._to_1d_array(self.Q).reshape(-1, 1)

        if P.shape[1] != len(assets):
            raise ValueError(
                f"P column count ({P.shape[1]}) must match number of assets ({len(assets)})."
            )
        if P.shape[0] != Q.shape[0]:
            raise ValueError(
                f"P row count ({P.shape[0]}) must match Q length ({Q.shape[0]})."
            )

        omega = self._get_omega(P=P, sigma=sigma)
        self.last_omega_ = pd.DataFrame(
            omega,
            index=[f"view_{i}" for i in range(P.shape[0])],
            columns=[f"view_{i}" for i in range(P.shape[0])],
        )

        tau_sigma = self.tau * sigma
        middle = np.linalg.inv(P @ tau_sigma @ P.T + omega)
        posterior = pi + tau_sigma @ P.T @ middle @ (Q - P @ pi)

        posterior = pd.Series(
            posterior.flatten(),
            index=assets,
            name="expected_return",
        )
        return posterior

    def _get_omega(self, P: np.ndarray, sigma: np.ndarray) -> np.ndarray:
        if self.Omega is not None:
            return self._to_2d_array(self.Omega)

        base = np.diag(np.diag(self.tau * P @ sigma @ P.T))

        if self.view_confidences is None:
            return base

        c = self._to_1d_array(self.view_confidences)
        if len(c) != P.shape[0]:
            raise ValueError("view_confidences length must match number of views.")

        c = np.clip(c.astype(float), 1e-6, 1.0)
        scale = (1.0 - c) / c
        omega = np.diag(np.diag(base) * scale)
        return omega

    def _to_2d_array(self, x) -> np.ndarray:
        if isinstance(x, pd.DataFrame):
            arr = x.values
        elif isinstance(x, pd.Series):
            arr = x.values.reshape(1, -1)
        else:
            arr = np.asarray(x)

        if arr.ndim != 2:
            raise ValueError("Input must be 2D.")
        return arr.astype(float)

    def _to_1d_array(self, x) -> np.ndarray:
        if isinstance(x, pd.Series):
            arr = x.values
        elif isinstance(x, pd.DataFrame):
            if x.shape[1] != 1:
                raise ValueError("1D DataFrame input must have exactly one column.")
            arr = x.iloc[:, 0].values
        else:
            arr = np.asarray(x)

        if arr.ndim != 1:
            raise ValueError("Input must be 1D.")
        return arr.astype(float)