from __future__ import annotations

"""
DQN_port — Double DQN model selector optimised for **portfolio Sharpe ratio**.

Reward : EMA-Sharpe of realised BL portfolio return
State  : cross-sectional prediction statistics (pre-computed externally)
Action : choose one of N forecast models for the entire cross-section
Method : Double DQN with replay buffer, LayerNorm, gamma=0.9

Usage
-----
    cfg = DQNPortConfig(episodes=25, gamma=0.9)
    selector = DQNPortSelector(model_names, config=cfg)
    log_df   = selector.fit(train_states, reward_panel)
    actions  = selector.predict_actions(test_states)
"""

import random
import time
from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd
import torch
from torch import nn, optim

from ._common import (
    QNetwork,
    ReplayBuffer,
    EMASharpe,
    DifferentialSharpe,
    RunningNormalizer,
    resolve_device,
    seed_everything,
)


@dataclass
class DQNPortConfig:
    episodes: int = 25
    gamma: float = 0.9               # regime memory
    learning_rate: float = 1e-3
    batch_size: int = 256
    hidden_dims: Sequence[int] = (256, 256)
    target_update_every: int = 200    # hard target sync (steps)
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay: float = 0.997      # per-episode
    replay_capacity: int = 50_000
    # Reward shaping
    reward_type: str = "dsr"          # "dsr" | "ema_sharpe" | "raw_return"
    ema_alpha: float = 0.1             # (for ema_sharpe)
    dsr_eta: float = 0.04              # (for dsr)
    reward_scale: float = 10.0
    reward_normalise: bool = True      # running z-score for stability
    risk_penalty: float = 0.0          # extra penalty on |return| (drawdown guard)
    horizon_days: int = 5
    periods_per_year: int = 252
    seed: int = 42
    device: str | None = None


class DQNPortSelector:
    """
    Double DQN model-selector for BL portfolio Sharpe maximisation.

    Inputs (from main experiment script):
      train_states : DataFrame[fold_id, date, state]
                     state = np.float32 array (pre-built, normalised)
      reward_panel : DataFrame[fold_id, date, reward_blret__model1, ...]
                     realised BL portfolio return per (fold, date, model)

    Double DQN update:
      target = r + γ · Q_target(s', argmax_a Q_online(s', a))
    """

    def __init__(self, model_names: list[str],
                 config: DQNPortConfig | None = None):
        self.model_names = list(model_names)
        self.cfg = config or DQNPortConfig()
        self.device = resolve_device(self.cfg.device)
        self._dev = torch.device(self.device)

        n = len(self.model_names)
        self.in_dim = n * 3 + 3      # pred_mean/abs/std per model + 3 global
        self.out_dim = n

        self.q_net = QNetwork(self.in_dim, self.out_dim, self.cfg.hidden_dims).to(self._dev)
        self.target_net = QNetwork(self.in_dim, self.out_dim, self.cfg.hidden_dims).to(self._dev)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.q_net.parameters(), lr=self.cfg.learning_rate)
        self.loss_fn = nn.SmoothL1Loss()
        self.replay = ReplayBuffer(self.cfg.replay_capacity)

        self._rng = seed_everything(self.cfg.seed, self.device)

    # -----------------------------------------------------------------
    def _greedy(self, state: np.ndarray) -> int:
        with torch.no_grad():
            x = torch.tensor(state, dtype=torch.float32, device=self._dev).unsqueeze(0)
            return int(self.q_net(x).argmax(1).item())

    def _act(self, state: np.ndarray, epsilon: float) -> int:
        if self._rng.random() < epsilon:
            return int(self._rng.integers(self.out_dim))
        return self._greedy(state)

    def _optimize(self) -> float | None:
        if len(self.replay) < self.cfg.batch_size:
            return None
        s, a, r, s2, d = self.replay.sample(self.cfg.batch_size, self._rng)

        S  = torch.tensor(s,  dtype=torch.float32, device=self._dev)
        A  = torch.tensor(a,  dtype=torch.int64,   device=self._dev).unsqueeze(1)
        R  = torch.tensor(r,  dtype=torch.float32, device=self._dev)
        S2 = torch.tensor(s2, dtype=torch.float32, device=self._dev)
        D  = torch.tensor(d,  dtype=torch.float32, device=self._dev)

        q_curr = self.q_net(S).gather(1, A).squeeze(1)

        with torch.no_grad():
            best_a = self.q_net(S2).argmax(1, keepdim=True)
            q_next = self.target_net(S2).gather(1, best_a).squeeze(1)
            target = R + (1.0 - D) * self.cfg.gamma * q_next

        loss = self.loss_fn(q_curr, target)
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.q_net.parameters(), max_norm=1.0)
        self.optimizer.step()
        return float(loss.item())

    # -----------------------------------------------------------------
    # fit
    # -----------------------------------------------------------------
    def fit(
        self, train_states: pd.DataFrame, reward_panel: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Train on view-build data.  EMA-Sharpe resets at each fold boundary.
        Returns training log DataFrame.
        """
        train_states = train_states.sort_values(["fold_id", "date"]).reset_index(drop=True)
        reward_panel = reward_panel.sort_values(["fold_id", "date"]).reset_index(drop=True)
        merged = train_states.merge(reward_panel, on=["fold_id", "date"], how="inner")
        if merged.empty:
            raise ValueError("Merged frame is empty — check fold_id/date alignment.")

        by_fold = list(merged.groupby("fold_id", sort=True))
        epsilon = self.cfg.epsilon_start
        global_step = 0
        rows: list[dict] = []

        for ep in range(1, self.cfg.episodes + 1):
            ep_losses: list[float] = []
            ep_rewards: list[float] = []
            t0 = time.time()

            for _, fold_df in by_fold:
                fold_df = fold_df.sort_values("date").reset_index(drop=True)
                states = np.stack(fold_df["state"].to_numpy())
                n = len(fold_df)
                done_arr = np.zeros(n, dtype=bool)
                if n > 0:
                    done_arr[-1] = True

                # Reward shaper (reset each fold)
                if self.cfg.reward_type == "dsr":
                    shaper = DifferentialSharpe(eta=self.cfg.dsr_eta)
                elif self.cfg.reward_type == "ema_sharpe":
                    shaper = EMASharpe(
                        alpha=self.cfg.ema_alpha,
                        horizon=self.cfg.horizon_days,
                        periods_per_year=self.cfg.periods_per_year,
                    )
                else:  # raw_return
                    shaper = None
                normaliser = RunningNormalizer() if self.cfg.reward_normalise else None

                for i in range(n):
                    state = states[i]
                    next_state = states[i + 1] if i + 1 < n else np.zeros_like(state)
                    action = self._act(state, epsilon)
                    model_name = self.model_names[action]

                    bl_ret = float(fold_df.loc[i, f"reward_blret__{model_name}"])
                    # Primary shaped reward
                    if shaper is not None:
                        base_reward = shaper.update(bl_ret)
                    else:
                        base_reward = bl_ret
                    # Risk penalty on |return| (discourages volatile picks)
                    base_reward -= self.cfg.risk_penalty * abs(bl_ret)
                    # Normalise to stationary z-score for stable learning
                    if normaliser is not None:
                        base_reward = normaliser.update(base_reward)
                    reward = base_reward * self.cfg.reward_scale
                    ep_rewards.append(reward)

                    self.replay.push(state, action, reward, next_state, bool(done_arr[i]))
                    loss = self._optimize()
                    if loss is not None:
                        ep_losses.append(loss)

                    global_step += 1
                    if global_step % self.cfg.target_update_every == 0:
                        self.target_net.load_state_dict(self.q_net.state_dict())

            epsilon = max(self.cfg.epsilon_end, epsilon * self.cfg.epsilon_decay)
            avg_l = float(np.mean(ep_losses)) if ep_losses else float("nan")
            avg_r = float(np.mean(ep_rewards)) if ep_rewards else float("nan")
            elapsed = time.time() - t0
            eta = (self.cfg.episodes - ep) * elapsed
            print(
                f"[DQN_port] ep={ep:3d}/{self.cfg.episodes} "
                f"eps={epsilon:.4f} loss={avg_l:.6f} reward={avg_r:.6f} "
                f"elapsed={elapsed/60:.1f}m eta={eta/60:.1f}m"
            )
            rows.append({"episode": ep, "epsilon": epsilon,
                         "avg_loss": avg_l, "avg_reward": avg_r})

        self.training_log_ = pd.DataFrame(rows)
        return self.training_log_

    # -----------------------------------------------------------------
    # predict_actions
    # -----------------------------------------------------------------
    def predict_actions(self, states: pd.DataFrame) -> pd.DataFrame:
        states = states.sort_values(["fold_id", "date"]).reset_index(drop=True).copy()
        if states.empty:
            return pd.DataFrame(columns=["fold_id", "date", "selected_action", "selected_model"])
        actions = [self._greedy(s) for s in states["state"].to_numpy()]
        out = states[["fold_id", "date"]].copy()
        out["selected_action"] = actions
        out["selected_model"] = [self.model_names[a] for a in actions]
        return out
