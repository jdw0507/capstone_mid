from __future__ import annotations

"""
PPO (Proximal Policy Optimization) model selector.

Same interface as DQNModelSelector:
  - fit(train_df, candidate_models)  → trains the agent
  - select_predictions(inference_df) → selects best model per (date, asset)

Key differences from DQN:
  - On-policy: collects full trajectories per epoch, then updates
  - Actor-Critic: shared backbone with policy head (Categorical) + value head
  - PPO clip objective: prevents destructively large policy updates
  - GAE (Generalised Advantage Estimation): lower-variance advantage signal
  - Entropy bonus: encourages exploration
"""

from collections import deque
from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.distributions import Categorical


# =============================================================================
# Diagnostics container
# =============================================================================
@dataclass
class PPOTrainingDiagnostics:
    fold_id: int
    n_samples: int
    n_assets: int
    n_models: int
    epochs: int
    steps: int
    avg_policy_loss: float
    avg_value_loss: float
    avg_entropy: float
    fallback_model: str | None


# =============================================================================
# Actor-Critic network
# =============================================================================
class _ActorCriticNetwork(nn.Module):
    """Shared backbone → policy head (logits) + value head (scalar)."""

    def __init__(
        self, input_dim: int, output_dim: int, hidden_dims: Sequence[int]
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        prev = input_dim
        for h in hidden_dims:
            layers.extend([nn.Linear(prev, h), nn.LayerNorm(h), nn.ReLU()])
            prev = h
        self.backbone = nn.Sequential(*layers)
        self.policy_head = nn.Linear(prev, output_dim)
        self.value_head = nn.Linear(prev, 1)

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.backbone(x)
        logits = self.policy_head(h)
        value = self.value_head(h).squeeze(-1)
        return logits, value

    def get_policy(self, x: torch.Tensor) -> torch.Tensor:
        return self.policy_head(self.backbone(x))

    def get_value(self, x: torch.Tensor) -> torch.Tensor:
        return self.value_head(self.backbone(x)).squeeze(-1)


# =============================================================================
# Trajectory buffer (on-policy)
# =============================================================================
class _TrajectoryBuffer:
    """Collects (s, a, r, done, log_prob, value) tuples for on-policy update."""

    def __init__(self) -> None:
        self.states: list[np.ndarray] = []
        self.actions: list[int] = []
        self.rewards: list[float] = []
        self.dones: list[bool] = []
        self.log_probs: list[float] = []
        self.values: list[float] = []

    def push(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        done: bool,
        log_prob: float,
        value: float,
    ) -> None:
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.dones.append(done)
        self.log_probs.append(log_prob)
        self.values.append(value)

    def __len__(self) -> int:
        return len(self.states)

    def clear(self) -> None:
        self.states.clear()
        self.actions.clear()
        self.rewards.clear()
        self.dones.clear()
        self.log_probs.clear()
        self.values.clear()

    def to_tensors(
        self, device: torch.device
    ) -> dict[str, torch.Tensor]:
        return {
            "states": torch.tensor(
                np.stack(self.states), dtype=torch.float32, device=device
            ),
            "actions": torch.tensor(
                self.actions, dtype=torch.int64, device=device
            ),
            "rewards": torch.tensor(
                self.rewards, dtype=torch.float32, device=device
            ),
            "dones": torch.tensor(
                self.dones, dtype=torch.float32, device=device
            ),
            "old_log_probs": torch.tensor(
                self.log_probs, dtype=torch.float32, device=device
            ),
            "old_values": torch.tensor(
                self.values, dtype=torch.float32, device=device
            ),
        }


# =============================================================================
# GAE computation
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
    Compute Generalised Advantage Estimation.

    Returns
    -------
    advantages : Tensor[T]
    returns    : Tensor[T]   (advantages + values, used as value target)
    """
    T = len(rewards)
    advantages = torch.zeros(T, device=rewards.device)
    gae = 0.0
    next_value = last_value

    for t in reversed(range(T)):
        mask = 1.0 - dones[t].item()
        delta = rewards[t].item() + gamma * next_value * mask - values[t].item()
        gae = delta + gamma * lam * mask * gae
        advantages[t] = gae
        next_value = values[t].item()

    returns = advantages + values
    return advantages, returns


# =============================================================================
# PPO model selector
# =============================================================================
class PPOModelSelector:
    """
    PPO agent for model-selection.
    State  : model prediction vector at (date, asset)
    Action : choose one model index
    Reward : -abs(pred - true)

    Same interface as DQNModelSelector — drop-in replacement.
    """

    def __init__(
        self,
        hidden_dims: Sequence[int] = (128, 64),
        gamma: float = 0.95,
        lam: float = 0.95,
        lr: float = 3e-4,
        clip_eps: float = 0.2,
        entropy_coef: float = 0.01,
        value_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        mini_batch_size: int = 256,
        ppo_update_epochs: int = 4,
        epochs: int = 12,
        random_state: int = 42,
        device: str | None = None,
        verbose: bool = True,
    ) -> None:
        self.hidden_dims = tuple(hidden_dims)
        self.gamma = float(gamma)
        self.lam = float(lam)
        self.lr = float(lr)
        self.clip_eps = float(clip_eps)
        self.entropy_coef = float(entropy_coef)
        self.value_coef = float(value_coef)
        self.max_grad_norm = float(max_grad_norm)
        self.mini_batch_size = int(mini_batch_size)
        self.ppo_update_epochs = int(ppo_update_epochs)
        self.epochs = int(epochs)
        self.random_state = int(random_state)
        self.verbose = verbose

        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        self.candidate_models_: list[str] | None = None
        self.model_fill_values_: pd.Series | None = None
        self.ac_net_: _ActorCriticNetwork | None = None
        self.optimizer_: torch.optim.Optimizer | None = None
        self.fallback_model_: str | None = None

        self._rng = np.random.default_rng(self.random_state)
        torch.manual_seed(self.random_state)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.random_state)

    # -----------------------------------------------------------------
    # Public API — same as DQNModelSelector
    # -----------------------------------------------------------------
    def fit(
        self,
        train_df: pd.DataFrame,
        candidate_models: Iterable[str],
    ) -> "PPOModelSelector":
        candidate_models = list(candidate_models)
        if len(candidate_models) < 2:
            raise ValueError(
                "candidate_models must have at least two models for PPO selection."
            )

        bundle = self._prepare_dataset(train_df, candidate_models, fit_mode=True)
        states = bundle["states"]
        rewards = bundle["rewards"]
        episodes = bundle["episodes"]

        # Fallback: static best model by MAE
        model_mae = (
            train_df[train_df["model"].isin(candidate_models)]
            .assign(abs_error=lambda x: (x["y_pred"] - x["y_true"]).abs())
            .groupby("model")["abs_error"]
            .mean()
            .sort_values()
        )
        self.fallback_model_ = (
            str(model_mae.index[0]) if not model_mae.empty else candidate_models[0]
        )

        if len(states) < 64 or len(episodes) == 0:
            if self.verbose:
                print(
                    f"[PPO] not enough samples (n={len(states)}). "
                    f"fallback to static model: {self.fallback_model_}"
                )
            self.candidate_models_ = candidate_models
            return self

        input_dim = states.shape[1]
        n_actions = len(candidate_models)

        self.ac_net_ = _ActorCriticNetwork(
            input_dim, n_actions, self.hidden_dims
        ).to(self.device)
        self.optimizer_ = torch.optim.Adam(self.ac_net_.parameters(), lr=self.lr)

        all_policy_losses: list[float] = []
        all_value_losses: list[float] = []
        all_entropies: list[float] = []

        for ep in range(self.epochs):
            # ── 1. Collect trajectories (on-policy) ──────────────────
            buffer = _TrajectoryBuffer()
            shuffled = episodes.copy()
            self._rng.shuffle(shuffled)

            with torch.no_grad():
                for seq in shuffled:
                    for pos, idx in enumerate(seq):
                        state_np = states[idx]
                        state_t = torch.tensor(
                            state_np, dtype=torch.float32, device=self.device
                        ).unsqueeze(0)

                        logits, value = self.ac_net_(state_t)
                        dist = Categorical(logits=logits.squeeze(0))
                        action = dist.sample()
                        log_prob = dist.log_prob(action)

                        reward = float(rewards[idx, action.item()])
                        done = pos == (len(seq) - 1)

                        buffer.push(
                            state=state_np,
                            action=action.item(),
                            reward=reward,
                            done=done,
                            log_prob=log_prob.item(),
                            value=value.squeeze(0).item(),
                        )

            if len(buffer) == 0:
                continue

            # ── 2. Compute GAE advantages ────────────────────────────
            data = buffer.to_tensors(torch.device(self.device))
            advantages, returns = compute_gae(
                rewards=data["rewards"],
                values=data["old_values"],
                dones=data["dones"],
                gamma=self.gamma,
                lam=self.lam,
            )
            # Normalise advantages
            adv_std = advantages.std()
            if adv_std > 1e-8:
                advantages = (advantages - advantages.mean()) / adv_std

            # ── 3. PPO mini-batch updates ────────────────────────────
            n = len(buffer)
            indices = np.arange(n)

            for _ in range(self.ppo_update_epochs):
                self._rng.shuffle(indices)
                for start in range(0, n, self.mini_batch_size):
                    end = min(start + self.mini_batch_size, n)
                    mb_idx = indices[start:end]

                    mb_states = data["states"][mb_idx]
                    mb_actions = data["actions"][mb_idx]
                    mb_old_lp = data["old_log_probs"][mb_idx]
                    mb_adv = advantages[mb_idx]
                    mb_ret = returns[mb_idx]

                    logits, values = self.ac_net_(mb_states)
                    dist = Categorical(logits=logits)
                    new_lp = dist.log_prob(mb_actions)
                    entropy = dist.entropy().mean()

                    # Policy loss (clipped surrogate)
                    ratio = torch.exp(new_lp - mb_old_lp)
                    surr1 = ratio * mb_adv
                    surr2 = (
                        torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps)
                        * mb_adv
                    )
                    policy_loss = -torch.min(surr1, surr2).mean()

                    # Value loss
                    value_loss = nn.functional.mse_loss(values, mb_ret)

                    # Combined loss
                    loss = (
                        policy_loss
                        + self.value_coef * value_loss
                        - self.entropy_coef * entropy
                    )

                    self.optimizer_.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(
                        self.ac_net_.parameters(), self.max_grad_norm
                    )
                    self.optimizer_.step()

                    all_policy_losses.append(policy_loss.item())
                    all_value_losses.append(value_loss.item())
                    all_entropies.append(entropy.item())

            if self.verbose:
                pl = float(np.mean(all_policy_losses[-50:])) if all_policy_losses else float("nan")
                vl = float(np.mean(all_value_losses[-50:])) if all_value_losses else float("nan")
                ent = float(np.mean(all_entropies[-50:])) if all_entropies else float("nan")
                print(
                    f"[PPO] epoch={ep + 1}/{self.epochs} "
                    f"policy_loss={pl:.6f} value_loss={vl:.6f} entropy={ent:.4f}"
                )

        self.candidate_models_ = candidate_models
        return self

    def select_predictions(
        self,
        inference_df: pd.DataFrame,
        output_model_name: str = "PPO_Selector",
    ) -> pd.DataFrame:
        self._check_fitted()
        candidate_models = list(self.candidate_models_)

        bundle = self._prepare_dataset(inference_df, candidate_models, fit_mode=False)
        states = bundle["states"]
        keys = bundle["keys"].copy()

        if len(states) == 0:
            return pd.DataFrame(
                columns=list(inference_df.columns) + ["selected_model", "selector_action_prob"]
            )

        if self.ac_net_ is None:
            chosen = np.full(len(states), self.fallback_model_, dtype=object)
            probs = np.full(len(states), np.nan, dtype=float)
        else:
            with torch.no_grad():
                logits = self.ac_net_.get_policy(
                    torch.tensor(states, dtype=torch.float32, device=self.device)
                )
                dist = Categorical(logits=logits)
                action_idx = dist.probs.argmax(dim=1).cpu().numpy()
                action_probs = (
                    dist.probs[torch.arange(len(action_idx)), action_idx].cpu().numpy()
                )
            chosen = np.asarray(
                [candidate_models[i] for i in action_idx], dtype=object
            )
            probs = action_probs

        keys["selected_model"] = chosen
        keys["selector_action_prob"] = probs

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
        extra_cols = [
            c for c in ["selected_model", "selector_action_prob"] if c in merged.columns
        ]
        return merged[base_cols + extra_cols].copy()

    # -----------------------------------------------------------------
    # Internal helpers — same data preparation logic as DQN
    # -----------------------------------------------------------------
    def _prepare_dataset(
        self,
        df: pd.DataFrame,
        candidate_models: list[str],
        fit_mode: bool,
    ) -> dict[str, object]:
        req = {"date", "asset", "model", "y_pred"}
        miss = req - set(df.columns)
        if miss:
            raise ValueError(f"Missing required columns for PPO selector: {miss}")

        work = df.copy()
        work["date"] = pd.to_datetime(work["date"])
        work = work[work["model"].isin(candidate_models)].copy()

        pred_wide = (
            work.pivot_table(
                index=["date", "asset"], columns="model", values="y_pred", aggfunc="mean"
            )
            .reindex(columns=candidate_models)
            .sort_index()
        )
        if pred_wide.empty:
            return {
                "states": np.empty((0, 0), dtype=np.float32),
                "rewards": np.empty((0, 0), dtype=np.float32),
                "episodes": [],
                "keys": pd.DataFrame(columns=["date", "asset"]),
            }

        if fit_mode:
            self.model_fill_values_ = pred_wide.median(axis=0).fillna(0.0)
        if self.model_fill_values_ is None:
            raise ValueError("model_fill_values_ is not fitted.")

        pred_filled = pred_wide.fillna(self.model_fill_values_)
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

        episode_ids: list[list[int]] = []
        key_sorted = keys.sort_values(["asset", "date"]).copy()
        for _, g in key_sorted.groupby("asset"):
            seq = g["sample_id"].astype(int).tolist()
            if len(seq) >= 2:
                episode_ids.append(seq)

        return {
            "states": state,
            "rewards": reward_mat,
            "episodes": episode_ids,
            "keys": keys[["date", "asset"]].copy(),
        }

    def _check_fitted(self) -> None:
        if self.candidate_models_ is None or self.model_fill_values_ is None:
            raise ValueError("PPOModelSelector is not fitted yet.")


# =============================================================================
# Convenience wrapper — same pattern as build_dqn_selected_predictions
# =============================================================================
def build_ppo_selected_predictions(
    per_prediction: pd.DataFrame,
    candidate_models: Sequence[str],
    train_split: str = "view_build",
    infer_splits: Sequence[str] = ("view_build", "test"),
    output_model_name: str = "PPO_Selector",
    ppo_params: dict | None = None,
    verbose: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Train PPO selector per fold on train_split and emit selected rows
    for infer_splits.  Same interface as build_dqn_selected_predictions.
    """
    if (
        "fold_id" not in per_prediction.columns
        or "evaluation_split" not in per_prediction.columns
    ):
        raise ValueError(
            "per_prediction must include fold_id and evaluation_split "
            "for fold-wise PPO fitting."
        )

    ppo_params = ppo_params or {}
    selected_parts: list[pd.DataFrame] = []
    diag_rows: list[dict] = []

    fold_ids = sorted(
        pd.to_numeric(per_prediction["fold_id"], errors="coerce")
        .dropna()
        .astype(int)
        .unique()
    )

    for fold_id in fold_ids:
        train_df = per_prediction[
            (per_prediction["fold_id"] == fold_id)
            & (per_prediction["evaluation_split"] == train_split)
            & (per_prediction["model"].isin(candidate_models))
        ].copy()

        if train_df.empty:
            continue

        selector_kwargs = dict(ppo_params)
        selector_kwargs.setdefault("verbose", verbose)
        selector = PPOModelSelector(**selector_kwargs)
        selector.fit(train_df=train_df, candidate_models=candidate_models)

        for split_name in infer_splits:
            infer_df = per_prediction[
                (per_prediction["fold_id"] == fold_id)
                & (per_prediction["evaluation_split"] == split_name)
                & (per_prediction["model"].isin(candidate_models))
            ].copy()
            if infer_df.empty:
                continue

            chosen = selector.select_predictions(
                infer_df, output_model_name=output_model_name
            )
            selected_parts.append(chosen)

        diag_rows.append(
            {
                "fold_id": fold_id,
                "n_train_rows": int(len(train_df)),
                "n_train_assets": (
                    int(train_df["asset"].nunique())
                    if "asset" in train_df.columns
                    else np.nan
                ),
                "n_candidate_models": int(len(candidate_models)),
                "fallback_model": selector.fallback_model_,
                "used_ppo": int(selector.ac_net_ is not None),
            }
        )

    selected_df = (
        pd.concat(selected_parts, axis=0, ignore_index=True)
        if selected_parts
        else pd.DataFrame()
    )
    diagnostics_df = pd.DataFrame(diag_rows)
    return selected_df, diagnostics_df
