from __future__ import annotations

"""
DQN_pred — DQN model selector optimised for **prediction accuracy**.

Reward : -|y_pred - y_true|  (per asset, per date)
State  : model prediction vectors + z-score normalised features
Action : choose one of N candidate forecast models
Method : Double DQN with replay buffer (off-policy)

Usage
-----
    selector = DQNPredSelector(hidden_dims=(128, 64), episodes=12)
    selector.fit(train_df, candidate_models)
    selected = selector.select_predictions(test_df, output_model_name="DQN_Pred")
"""

from collections import deque
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
import torch
from torch import nn

from ._common import (
    QNetwork,
    ReplayBuffer,
    prepare_pred_dataset,
    merge_selected_predictions,
    resolve_device,
    seed_everything,
)


class DQNPredSelector:
    """
    Double DQN agent that selects the forecast model with the lowest
    prediction error for each (date, asset) pair.
    """

    def __init__(
        self,
        hidden_dims: Sequence[int] = (128, 64),
        gamma: float = 0.95,
        lr: float = 1e-3,
        batch_size: int = 128,
        replay_capacity: int = 50_000,
        min_replay_size: int = 512,
        target_update_interval: int = 200,
        episodes: int = 12,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.05,
        epsilon_decay: float = 0.995,
        random_state: int = 42,
        device: str | None = None,
        verbose: bool = True,
    ) -> None:
        self.hidden_dims = tuple(hidden_dims)
        self.gamma = float(gamma)
        self.lr = float(lr)
        self.batch_size = int(batch_size)
        self.replay_capacity = int(replay_capacity)
        self.min_replay_size = int(min_replay_size)
        self.target_update_interval = int(target_update_interval)
        self.episodes = int(episodes)
        self.epsilon_start = float(epsilon_start)
        self.epsilon_end = float(epsilon_end)
        self.epsilon_decay = float(epsilon_decay)
        self.random_state = int(random_state)
        self.device = resolve_device(device)
        self.verbose = verbose

        self.candidate_models_: list[str] | None = None
        self.model_fill_values_: pd.Series | None = None
        self.q_net_: QNetwork | None = None
        self.target_net_: QNetwork | None = None
        self.optimizer_: torch.optim.Optimizer | None = None
        self.fallback_model_: str | None = None

        self._rng = seed_everything(self.random_state, self.device)

    # -----------------------------------------------------------------
    # fit
    # -----------------------------------------------------------------
    def fit(
        self,
        train_df: pd.DataFrame,
        candidate_models: Iterable[str],
    ) -> "DQNPredSelector":
        candidate_models = list(candidate_models)
        if len(candidate_models) < 2:
            raise ValueError("Need at least 2 candidate models.")

        bundle, new_fill = prepare_pred_dataset(
            train_df, candidate_models, None, fit_mode=True,
        )
        self.model_fill_values_ = new_fill
        states = bundle["states"]
        rewards = bundle["rewards"]
        episodes = bundle["episodes"]

        # Fallback: static best model by MAE
        model_mae = (
            train_df[train_df["model"].isin(candidate_models)]
            .assign(ae=lambda x: (x["y_pred"] - x["y_true"]).abs())
            .groupby("model")["ae"].mean().sort_values()
        )
        self.fallback_model_ = (
            str(model_mae.index[0]) if not model_mae.empty else candidate_models[0]
        )

        if len(states) < 64 or not episodes:
            if self.verbose:
                print(f"[DQN_pred] Not enough data (n={len(states)}), "
                      f"fallback → {self.fallback_model_}")
            self.candidate_models_ = candidate_models
            return self

        n_actions = len(candidate_models)
        input_dim = states.shape[1]
        dev = torch.device(self.device)

        self.q_net_ = QNetwork(input_dim, n_actions, self.hidden_dims).to(dev)
        self.target_net_ = QNetwork(input_dim, n_actions, self.hidden_dims).to(dev)
        self.target_net_.load_state_dict(self.q_net_.state_dict())
        self.target_net_.eval()
        self.optimizer_ = torch.optim.Adam(self.q_net_.parameters(), lr=self.lr)

        replay = ReplayBuffer(self.replay_capacity)
        epsilon = self.epsilon_start
        global_step = 0
        losses: list[float] = []

        for ep in range(self.episodes):
            shuffled = episodes.copy()
            self._rng.shuffle(shuffled)

            for seq in shuffled:
                for pos, idx in enumerate(seq):
                    state = states[idx]

                    # Epsilon-greedy action
                    if self._rng.random() < epsilon:
                        action = int(self._rng.integers(0, n_actions))
                    else:
                        with torch.no_grad():
                            q = self.q_net_(
                                torch.tensor(state, dtype=torch.float32, device=dev).unsqueeze(0)
                            )
                            action = int(q.argmax(1).item())

                    reward = float(rewards[idx, action])
                    done = pos == len(seq) - 1
                    next_state = (
                        np.zeros_like(state, dtype=np.float32) if done
                        else states[seq[pos + 1]]
                    )

                    replay.push(state, action, reward, next_state, done)

                    if len(replay) >= self.min_replay_size:
                        loss = self._train_step(replay, dev)
                        if np.isfinite(loss):
                            losses.append(loss)

                    global_step += 1
                    if global_step % self.target_update_interval == 0:
                        self.target_net_.load_state_dict(self.q_net_.state_dict())

            epsilon = max(self.epsilon_end, epsilon * self.epsilon_decay)
            if self.verbose:
                avg = float(np.mean(losses[-200:])) if losses else float("nan")
                print(f"[DQN_pred] ep={ep+1}/{self.episodes} "
                      f"eps={epsilon:.4f} loss={avg:.6f}")

        self.candidate_models_ = candidate_models
        return self

    # -----------------------------------------------------------------
    # select_predictions
    # -----------------------------------------------------------------
    def select_predictions(
        self,
        inference_df: pd.DataFrame,
        output_model_name: str = "DQN_Pred",
    ) -> pd.DataFrame:
        self._check_fitted()
        candidate_models = list(self.candidate_models_)
        bundle, _ = prepare_pred_dataset(
            inference_df, candidate_models, self.model_fill_values_, fit_mode=False,
        )
        states = bundle["states"]
        keys = bundle["keys"]

        if len(states) == 0:
            return pd.DataFrame(
                columns=list(inference_df.columns) + ["selected_model", "selector_q_value"]
            )

        if self.q_net_ is None:
            chosen = np.full(len(states), self.fallback_model_, dtype=object)
            qvals = np.full(len(states), np.nan, dtype=float)
        else:
            dev = torch.device(self.device)
            with torch.no_grad():
                q = self.q_net_(
                    torch.tensor(states, dtype=torch.float32, device=dev)
                ).cpu().numpy()
            action_idx = np.argmax(q, axis=1)
            chosen = np.array([candidate_models[i] for i in action_idx], dtype=object)
            qvals = q[np.arange(len(q)), action_idx]

        return merge_selected_predictions(
            keys, chosen, {"selector_q_value": qvals},
            inference_df, output_model_name,
        )

    # -----------------------------------------------------------------
    # Double DQN training step
    # -----------------------------------------------------------------
    def _train_step(self, replay: ReplayBuffer, dev: torch.device) -> float:
        s, a, r, s2, d = replay.sample(self.batch_size, self._rng)
        S = torch.tensor(s, dtype=torch.float32, device=dev)
        A = torch.tensor(a, dtype=torch.int64, device=dev).unsqueeze(1)
        R = torch.tensor(r, dtype=torch.float32, device=dev).unsqueeze(1)
        S2 = torch.tensor(s2, dtype=torch.float32, device=dev)
        D = torch.tensor(d, dtype=torch.float32, device=dev).unsqueeze(1)

        q_pred = self.q_net_(S).gather(1, A)

        with torch.no_grad():
            # Double DQN: online selects, target evaluates
            best_a = self.q_net_(S2).argmax(1, keepdim=True)
            q_next = self.target_net_(S2).gather(1, best_a)
            q_target = R + (1.0 - D) * self.gamma * q_next

        loss = nn.functional.smooth_l1_loss(q_pred, q_target)
        self.optimizer_.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.q_net_.parameters(), max_norm=5.0)
        self.optimizer_.step()
        return float(loss.item())

    def _check_fitted(self) -> None:
        if self.candidate_models_ is None or self.model_fill_values_ is None:
            raise ValueError("DQNPredSelector not fitted.")


# =============================================================================
# Convenience wrapper (fold-wise train + infer)
# =============================================================================
def build_dqn_pred_selected_predictions(
    per_prediction: pd.DataFrame,
    candidate_models: Sequence[str],
    train_split: str = "view_build",
    infer_splits: Sequence[str] = ("view_build", "test"),
    output_model_name: str = "DQN_Pred",
    params: dict | None = None,
    verbose: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Train DQN_pred per fold, emit selected predictions."""
    if not {"fold_id", "evaluation_split"}.issubset(per_prediction.columns):
        raise ValueError("Need fold_id and evaluation_split columns.")

    params = params or {}
    parts: list[pd.DataFrame] = []
    diags: list[dict] = []

    for fold_id in sorted(per_prediction["fold_id"].dropna().unique()):
        train = per_prediction[
            (per_prediction["fold_id"] == fold_id)
            & (per_prediction["evaluation_split"] == train_split)
            & (per_prediction["model"].isin(candidate_models))
        ].copy()
        if train.empty:
            continue

        sel = DQNPredSelector(**{**params, "verbose": verbose})
        sel.fit(train, candidate_models)

        for split in infer_splits:
            infer = per_prediction[
                (per_prediction["fold_id"] == fold_id)
                & (per_prediction["evaluation_split"] == split)
                & (per_prediction["model"].isin(candidate_models))
            ].copy()
            if not infer.empty:
                parts.append(sel.select_predictions(infer, output_model_name))

        diags.append({
            "fold_id": fold_id,
            "n_train": len(train),
            "fallback_model": sel.fallback_model_,
            "used_dqn": int(sel.q_net_ is not None),
        })

    selected = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    return selected, pd.DataFrame(diags)
