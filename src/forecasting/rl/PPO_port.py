from __future__ import annotations

"""
PPO_port — PPO model selector optimised for **portfolio Sharpe ratio**.

Reward : EMA-Sharpe of realised BL portfolio return
State  : cross-sectional prediction statistics (pre-computed externally)
Action : choose one of N forecast models for the entire cross-section
Method : PPO (on-policy, Actor-Critic, clipped surrogate, GAE)

Usage
-----
    cfg = PPOPortConfig(epochs=25, gamma=0.9)
    selector = PPOPortSelector(model_names, config=cfg)
    log_df   = selector.fit(train_states, reward_panel)
    actions  = selector.predict_actions(test_states)
"""

import time
from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.distributions import Categorical

from ._common import (
    ActorCriticNetwork,
    TrajectoryBuffer,
    EMASharpe,
    DifferentialSharpe,
    RunningNormalizer,
    compute_gae,
    resolve_device,
    seed_everything,
)


@dataclass
class PPOPortConfig:
    epochs: int = 25
    gamma: float = 0.9               # regime memory
    lam: float = 0.95                # GAE lambda
    learning_rate: float = 3e-4
    clip_eps: float = 0.2
    entropy_coef: float = 0.01
    value_coef: float = 0.5
    max_grad_norm: float = 0.5
    hidden_dims: Sequence[int] = (256, 256)
    mini_batch_size: int = 256
    ppo_update_epochs: int = 4       # K epochs per trajectory collection
    # Reward shaping
    reward_type: str = "dsr"          # "dsr" | "ema_sharpe" | "raw_return"
    ema_alpha: float = 0.1             # (for ema_sharpe)
    dsr_eta: float = 0.04              # (for dsr)
    reward_scale: float = 10.0
    reward_normalise: bool = True      # running z-score for stability
    risk_penalty: float = 0.0          # penalty on |return|
    horizon_days: int = 5
    periods_per_year: int = 252
    seed: int = 42
    device: str | None = None


class PPOPortSelector:
    """
    PPO model-selector for BL portfolio Sharpe maximisation.

    Inputs (from main experiment script):
      train_states : DataFrame[fold_id, date, state]
      reward_panel : DataFrame[fold_id, date, reward_blret__model1, ...]

    PPO update:
      L_clip = min(r(θ)·A, clip(r(θ), 1±ε)·A)
      L      = -L_clip + c_v·L_value - c_e·H[π]
    """

    def __init__(self, model_names: list[str],
                 config: PPOPortConfig | None = None):
        self.model_names = list(model_names)
        self.cfg = config or PPOPortConfig()
        self.device = resolve_device(self.cfg.device)
        self._dev = torch.device(self.device)

        n = len(self.model_names)
        self.in_dim = n * 3 + 3
        self.out_dim = n

        self.ac_net = ActorCriticNetwork(
            self.in_dim, self.out_dim, self.cfg.hidden_dims
        ).to(self._dev)
        self.optimizer = torch.optim.Adam(
            self.ac_net.parameters(), lr=self.cfg.learning_rate
        )

        self._rng = seed_everything(self.cfg.seed, self.device)

    # -----------------------------------------------------------------
    # fit
    # -----------------------------------------------------------------
    def fit(
        self, train_states: pd.DataFrame, reward_panel: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Train on view-build data.  EMA-Sharpe resets at each fold boundary.
        Each epoch: collect full trajectory → compute GAE → K PPO updates.
        Returns training log DataFrame.
        """
        train_states = train_states.sort_values(["fold_id", "date"]).reset_index(drop=True)
        reward_panel = reward_panel.sort_values(["fold_id", "date"]).reset_index(drop=True)
        merged = train_states.merge(reward_panel, on=["fold_id", "date"], how="inner")
        if merged.empty:
            raise ValueError("Merged frame is empty — check fold_id/date alignment.")

        by_fold = list(merged.groupby("fold_id", sort=True))
        rows: list[dict] = []

        for ep in range(1, self.cfg.epochs + 1):
            t0 = time.time()

            # ── 1. Collect trajectory across all folds ───────────────
            buf = TrajectoryBuffer()

            with torch.no_grad():
                for _, fold_df in by_fold:
                    fold_df = fold_df.sort_values("date").reset_index(drop=True)
                    states_arr = np.stack(fold_df["state"].to_numpy())
                    n = len(fold_df)

                    # Reward shaper (reset each fold)
                    if self.cfg.reward_type == "dsr":
                        shaper = DifferentialSharpe(eta=self.cfg.dsr_eta)
                    elif self.cfg.reward_type == "ema_sharpe":
                        shaper = EMASharpe(
                            alpha=self.cfg.ema_alpha,
                            horizon=self.cfg.horizon_days,
                            periods_per_year=self.cfg.periods_per_year,
                        )
                    else:
                        shaper = None
                    normaliser = RunningNormalizer() if self.cfg.reward_normalise else None

                    for i in range(n):
                        s_t = torch.tensor(
                            states_arr[i], dtype=torch.float32, device=self._dev
                        ).unsqueeze(0)

                        logits, val = self.ac_net(s_t)
                        dist = Categorical(logits=logits.squeeze(0))
                        action = dist.sample()
                        log_prob = dist.log_prob(action)

                        model_name = self.model_names[action.item()]
                        bl_ret = float(fold_df.loc[i, f"reward_blret__{model_name}"])
                        if shaper is not None:
                            base_reward = shaper.update(bl_ret)
                        else:
                            base_reward = bl_ret
                        base_reward -= self.cfg.risk_penalty * abs(bl_ret)
                        if normaliser is not None:
                            base_reward = normaliser.update(base_reward)
                        reward = base_reward * self.cfg.reward_scale

                        done = (i == n - 1)
                        buf.push(
                            state=states_arr[i],
                            action=action.item(),
                            reward=reward,
                            done=done,
                            log_prob=log_prob.item(),
                            value=val.squeeze(0).item(),
                        )

            if len(buf) == 0:
                continue

            # ── 2. Compute GAE ───────────────────────────────────────
            data = buf.to_tensors(self._dev)
            adv, ret = compute_gae(
                data["rewards"], data["old_values"], data["dones"],
                self.cfg.gamma, self.cfg.lam,
            )
            std = adv.std()
            if std > 1e-8:
                adv = (adv - adv.mean()) / std

            # ── 3. PPO mini-batch updates (K epochs) ─────────────────
            n_samples = len(buf)
            indices = np.arange(n_samples)
            ep_pl: list[float] = []
            ep_vl: list[float] = []
            ep_ent: list[float] = []

            for _ in range(self.cfg.ppo_update_epochs):
                self._rng.shuffle(indices)
                for start in range(0, n_samples, self.cfg.mini_batch_size):
                    mb = indices[start: start + self.cfg.mini_batch_size]

                    mb_s = data["states"][mb]
                    mb_a = data["actions"][mb]
                    mb_olp = data["old_log_probs"][mb]
                    mb_adv = adv[mb]
                    mb_ret = ret[mb]

                    logits, vals = self.ac_net(mb_s)
                    dist = Categorical(logits=logits)
                    nlp = dist.log_prob(mb_a)
                    ent = dist.entropy().mean()

                    ratio = torch.exp(nlp - mb_olp)
                    s1 = ratio * mb_adv
                    s2 = torch.clamp(
                        ratio, 1 - self.cfg.clip_eps, 1 + self.cfg.clip_eps
                    ) * mb_adv
                    pl = -torch.min(s1, s2).mean()
                    vl = nn.functional.mse_loss(vals, mb_ret)

                    loss = pl + self.cfg.value_coef * vl - self.cfg.entropy_coef * ent

                    self.optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(
                        self.ac_net.parameters(), self.cfg.max_grad_norm
                    )
                    self.optimizer.step()

                    ep_pl.append(pl.item())
                    ep_vl.append(vl.item())
                    ep_ent.append(ent.item())

            avg_pl = float(np.mean(ep_pl)) if ep_pl else float("nan")
            avg_vl = float(np.mean(ep_vl)) if ep_vl else float("nan")
            avg_ent = float(np.mean(ep_ent)) if ep_ent else float("nan")
            elapsed = time.time() - t0
            eta = (self.cfg.epochs - ep) * elapsed
            print(
                f"[PPO_port] ep={ep:3d}/{self.cfg.epochs} "
                f"pl={avg_pl:.6f} vl={avg_vl:.6f} ent={avg_ent:.4f} "
                f"elapsed={elapsed/60:.1f}m eta={eta/60:.1f}m"
            )
            rows.append({
                "episode": ep,
                "avg_policy_loss": avg_pl,
                "avg_value_loss": avg_vl,
                "avg_entropy": avg_ent,
            })

        self.training_log_ = pd.DataFrame(rows)
        return self.training_log_

    # -----------------------------------------------------------------
    # predict_actions
    # -----------------------------------------------------------------
    def predict_actions(self, states: pd.DataFrame) -> pd.DataFrame:
        states = states.sort_values(["fold_id", "date"]).reset_index(drop=True).copy()
        if states.empty:
            return pd.DataFrame(
                columns=["fold_id", "date", "selected_action", "selected_model"]
            )
        with torch.no_grad():
            all_states = np.stack(states["state"].to_numpy())
            logits = self.ac_net.get_policy(
                torch.tensor(all_states, dtype=torch.float32, device=self._dev)
            )
            actions = logits.argmax(dim=1).cpu().numpy()

        out = states[["fold_id", "date"]].copy()
        out["selected_action"] = actions
        out["selected_model"] = [self.model_names[a] for a in actions]
        return out
