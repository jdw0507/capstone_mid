from __future__ import annotations

"""
Shared components for all RL model-selectors.

Networks  : QNetwork, ActorCriticNetwork
Buffers   : ReplayBuffer (off-policy), TrajectoryBuffer (on-policy)
Rewards   : EMASharpe (portfolio Sharpe estimator)
Advantage : compute_gae (Generalised Advantage Estimation)
Data      : prepare_pred_dataset (shared by _pred selectors)
"""

from typing import Sequence

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.distributions import Categorical


# =============================================================================
# Networks
# =============================================================================
class QNetwork(nn.Module):
    """MLP Q-network with LayerNorm for training stability."""

    def __init__(self, input_dim: int, output_dim: int,
                 hidden_dims: Sequence[int] = (256, 256)) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        prev = input_dim
        for h in hidden_dims:
            layers.extend([nn.Linear(prev, h), nn.LayerNorm(h), nn.ReLU()])
            prev = h
        layers.append(nn.Linear(prev, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ActorCriticNetwork(nn.Module):
    """Shared backbone → policy head (logits) + value head (scalar)."""

    def __init__(self, input_dim: int, output_dim: int,
                 hidden_dims: Sequence[int] = (256, 256)) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        prev = input_dim
        for h in hidden_dims:
            layers.extend([nn.Linear(prev, h), nn.LayerNorm(h), nn.ReLU()])
            prev = h
        self.backbone = nn.Sequential(*layers)
        self.policy_head = nn.Linear(prev, output_dim)
        self.value_head = nn.Linear(prev, 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.backbone(x)
        return self.policy_head(h), self.value_head(h).squeeze(-1)

    def get_policy(self, x: torch.Tensor) -> torch.Tensor:
        return self.policy_head(self.backbone(x))

    def get_value(self, x: torch.Tensor) -> torch.Tensor:
        return self.value_head(self.backbone(x)).squeeze(-1)


# =============================================================================
# Replay buffer (off-policy, for DQN)
# =============================================================================
class ReplayBuffer:
    def __init__(self, capacity: int):
        self._buf: list = []
        self._pos = 0
        self.capacity = capacity

    def __len__(self) -> int:
        return len(self._buf)

    def push(self, s: np.ndarray, a: int, r: float,
             s2: np.ndarray, done: bool) -> None:
        item = (s, a, r, s2, done)
        if len(self._buf) < self.capacity:
            self._buf.append(item)
        else:
            self._buf[self._pos] = item
        self._pos = (self._pos + 1) % self.capacity

    def sample(self, n: int, rng: np.random.Generator | None = None):
        import random as _random
        if rng is not None:
            idx = rng.choice(len(self._buf), size=min(n, len(self._buf)), replace=False)
            batch = [self._buf[i] for i in idx]
        else:
            batch = _random.sample(self._buf, min(n, len(self._buf)))
        s, a, r, s2, d = zip(*batch)
        return (
            np.stack(s),
            np.asarray(a, dtype=np.int64),
            np.asarray(r, dtype=np.float32),
            np.stack(s2),
            np.asarray(d, dtype=np.float32),
        )


# =============================================================================
# Trajectory buffer (on-policy, for PPO)
# =============================================================================
class TrajectoryBuffer:
    """Collects (s, a, r, done, log_prob, value) for on-policy update."""

    def __init__(self) -> None:
        self.states: list[np.ndarray] = []
        self.actions: list[int] = []
        self.rewards: list[float] = []
        self.dones: list[bool] = []
        self.log_probs: list[float] = []
        self.values: list[float] = []

    def __len__(self) -> int:
        return len(self.states)

    def push(self, state: np.ndarray, action: int, reward: float,
             done: bool, log_prob: float, value: float) -> None:
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.dones.append(done)
        self.log_probs.append(log_prob)
        self.values.append(value)

    def clear(self) -> None:
        for lst in (self.states, self.actions, self.rewards,
                    self.dones, self.log_probs, self.values):
            lst.clear()

    def to_tensors(self, device: torch.device) -> dict[str, torch.Tensor]:
        return {
            "states":        torch.tensor(np.stack(self.states), dtype=torch.float32, device=device),
            "actions":       torch.tensor(self.actions,          dtype=torch.int64,   device=device),
            "rewards":       torch.tensor(self.rewards,          dtype=torch.float32, device=device),
            "dones":         torch.tensor(self.dones,            dtype=torch.float32, device=device),
            "old_log_probs": torch.tensor(self.log_probs,        dtype=torch.float32, device=device),
            "old_values":    torch.tensor(self.values,           dtype=torch.float32, device=device),
        }


# =============================================================================
# GAE (Generalised Advantage Estimation)
# =============================================================================
def compute_gae(
    rewards: torch.Tensor,
    values: torch.Tensor,
    dones: torch.Tensor,
    gamma: float,
    lam: float,
    last_value: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Returns (advantages, returns).
    returns = advantages + values  (used as value target).
    """
    T = len(rewards)
    advantages = torch.zeros(T, device=rewards.device)
    gae = 0.0
    nv = last_value
    for t in reversed(range(T)):
        mask = 1.0 - dones[t].item()
        delta = rewards[t].item() + gamma * nv * mask - values[t].item()
        gae = delta + gamma * lam * mask * gae
        advantages[t] = gae
        nv = values[t].item()
    return advantages, advantages + values


# =============================================================================
# Differential Sharpe Ratio (Moody & Wu 1997) — better RL reward
# =============================================================================
class DifferentialSharpe:
    """
    Incremental Sharpe reward via the differential Sharpe ratio.

    Unlike EMA-Sharpe which returns the full Sharpe each step (large noisy scale),
    DSR returns the *first derivative* of Sharpe wrt each new return. This gives
    a bounded, well-scaled reward signal that is stationary and easier to learn.

    Formula (Moody & Wu 1997):
        A_t = (1 - eta) A_{t-1} + eta * r_t           (EMA of returns)
        B_t = (1 - eta) B_{t-1} + eta * r_t^2         (EMA of squared returns)
        D_t = [B_{t-1} * (r_t - A_{t-1}) - 0.5 * A_{t-1} * (r_t^2 - B_{t-1})]
              / (B_{t-1} - A_{t-1}^2)^(3/2)

    Returns are typically in [-1, +1] range for normal returns, giving stable
    RL learning vs EMA-Sharpe's unbounded scale.
    """

    def __init__(self, eta: float = 0.04):
        self.eta = eta
        self._A = 0.0       # EMA of returns
        self._B = 1e-6      # EMA of squared returns (tiny init for stability)
        self._init = False

    def reset(self) -> None:
        self._A = 0.0
        self._B = 1e-6
        self._init = False

    def update(self, r: float) -> float:
        r = float(r)
        if not self._init:
            # Warm-up: no reward on first step (need history)
            self._A = r
            self._B = r * r + 1e-6
            self._init = True
            return 0.0

        A_prev, B_prev = self._A, self._B
        denom = (B_prev - A_prev ** 2) ** 1.5
        if denom < 1e-10:
            # Degenerate (no variance) — fall back to raw return
            dsr = r
        else:
            num = B_prev * (r - A_prev) - 0.5 * A_prev * (r * r - B_prev)
            dsr = num / denom

        # Update EMAs for next step
        a = self.eta
        self._A = (1 - a) * A_prev + a * r
        self._B = (1 - a) * B_prev + a * r * r

        return float(dsr)


class RunningNormalizer:
    """Welford's online variance → reward normalisation (zero mean, unit std)."""

    def __init__(self, eps: float = 1e-8):
        self.n = 0
        self.mean = 0.0
        self.M2 = 0.0
        self.eps = eps

    def update(self, x: float) -> float:
        """Update stats and return z-score of x."""
        x = float(x)
        self.n += 1
        delta = x - self.mean
        self.mean += delta / self.n
        delta2 = x - self.mean
        self.M2 += delta * delta2
        if self.n < 2:
            return 0.0
        var = max(self.M2 / (self.n - 1), self.eps)
        return (x - self.mean) / var ** 0.5


# =============================================================================
# EMA-Sharpe reward (for _port selectors) — legacy, kept for backward compat
# =============================================================================
class EMASharpe:
    """
    Incremental Sharpe estimator via EMA of return and squared return.
    No warm-up: every step produces a non-zero reward signal.
    """

    def __init__(self, alpha: float = 0.1,
                 horizon: int = 5, periods_per_year: int = 252):
        self.alpha = alpha
        self.annualise = np.sqrt(periods_per_year / horizon)
        self._ema_r = 0.0
        self._ema_r2 = 0.0
        self._init = False

    def reset(self) -> None:
        self._ema_r = 0.0
        self._ema_r2 = 0.0
        self._init = False

    def update(self, r: float) -> float:
        if not self._init:
            self._ema_r = r
            self._ema_r2 = r * r
            self._init = True
        else:
            a = self.alpha
            self._ema_r = a * r + (1 - a) * self._ema_r
            self._ema_r2 = a * r * r + (1 - a) * self._ema_r2
        var = max(self._ema_r2 - self._ema_r ** 2, 1e-10)
        return float(self._ema_r / np.sqrt(var) * self.annualise)


# =============================================================================
# Shared data preparation (for _pred selectors)
# =============================================================================
def prepare_pred_dataset(
    df: pd.DataFrame,
    candidate_models: list[str],
    model_fill_values: pd.Series | None,
    fit_mode: bool,
) -> tuple[dict, pd.Series | None]:
    """
    Prepare state/reward matrices from per-prediction DataFrame.
    Used by both DQN_pred and PPO_pred.

    Returns
    -------
    bundle : dict with keys
        states   : np.ndarray (N, D)
        rewards  : np.ndarray (N, n_models)
        episodes : list[list[int]]
        keys     : pd.DataFrame [date, asset]
    new_fill_values : pd.Series | None  (only set when fit_mode=True)
    """
    req = {"date", "asset", "model", "y_pred"}
    miss = req - set(df.columns)
    if miss:
        raise ValueError(f"Missing columns: {miss}")

    work = df.copy()
    work["date"] = pd.to_datetime(work["date"])
    work = work[work["model"].isin(candidate_models)].copy()

    pred_wide = (
        work.pivot_table(index=["date", "asset"], columns="model",
                         values="y_pred", aggfunc="mean")
        .reindex(columns=candidate_models)
        .sort_index()
    )
    empty = {
        "states": np.empty((0, 0), dtype=np.float32),
        "rewards": np.empty((0, 0), dtype=np.float32),
        "episodes": [],
        "keys": pd.DataFrame(columns=["date", "asset"]),
    }
    if pred_wide.empty:
        return empty, model_fill_values

    new_fill = None
    if fit_mode:
        new_fill = pred_wide.median(axis=0).fillna(0.0)
        fill = new_fill
    else:
        if model_fill_values is None:
            raise ValueError("model_fill_values not set; call fit first.")
        fill = model_fill_values

    pred_filled = pred_wide.fillna(fill)
    pred_vals = pred_filled.to_numpy(dtype=np.float32)

    row_mean = pred_vals.mean(axis=1, keepdims=True)
    row_std = pred_vals.std(axis=1, keepdims=True)
    row_std = np.where(row_std <= 1e-8, 1.0, row_std)
    z = (pred_vals - row_mean) / row_std

    state = np.concatenate(
        [pred_vals, z, row_mean.astype(np.float32), row_std.astype(np.float32)],
        axis=1,
    ).astype(np.float32)

    keys = pred_filled.reset_index()[["date", "asset"]].copy()
    keys["sample_id"] = np.arange(len(keys), dtype=int)

    # Reward: -|pred - true| for each model
    reward_mat = np.zeros((len(keys), len(candidate_models)), dtype=np.float32)
    if "y_true" in work.columns:
        true_s = (
            work.dropna(subset=["y_true"])
            .groupby(["date", "asset"])["y_true"]
            .mean()
            .reindex(pred_wide.index)
        )
        true_vals = true_s.to_numpy(dtype=np.float32)
        err = np.abs(pred_wide.to_numpy(dtype=np.float32) - true_vals[:, None])
        finite_err = np.where(np.isfinite(err), err, np.nan)
        fallback = np.nanmax(finite_err, axis=1, keepdims=True)
        fallback = np.where(np.isfinite(fallback), fallback + 1.0, 10.0)
        err = np.where(np.isfinite(err), err, fallback)
        reward_mat = -err.astype(np.float32)

    # Episodes: sequential (date, asset) grouped by asset
    episode_ids: list[list[int]] = []
    key_sorted = keys.sort_values(["asset", "date"]).copy()
    for _, g in key_sorted.groupby("asset"):
        seq = g["sample_id"].astype(int).tolist()
        if len(seq) >= 2:
            episode_ids.append(seq)

    bundle = {
        "states": state,
        "rewards": reward_mat,
        "episodes": episode_ids,
        "keys": keys[["date", "asset"]].copy(),
    }
    return bundle, new_fill if fit_mode else model_fill_values


def merge_selected_predictions(
    keys: pd.DataFrame,
    chosen_models: np.ndarray,
    selector_meta: dict[str, np.ndarray],
    inference_df: pd.DataFrame,
    output_model_name: str,
) -> pd.DataFrame:
    """Merge selected model predictions back into the original DataFrame format."""
    keys = keys.copy()
    keys["selected_model"] = chosen_models
    for k, v in selector_meta.items():
        keys[k] = v

    merged = keys.merge(
        inference_df,
        left_on=["date", "asset", "selected_model"],
        right_on=["date", "asset", "model"],
        how="left",
        suffixes=("", "_raw"),
    )
    merged["model"] = output_model_name
    merged["selected_model"] = merged["selected_model"].astype(str)

    base_cols = list(inference_df.columns)
    extra_cols = [c for c in ["selected_model"] + list(selector_meta.keys())
                  if c in merged.columns]
    return merged[base_cols + extra_cols].copy()


def resolve_device(device: str | None) -> str:
    if device is not None:
        return device
    return "cuda" if torch.cuda.is_available() else "cpu"


def seed_everything(seed: int, device: str) -> np.random.Generator:
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device == "cuda" and torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    return np.random.default_rng(seed)
