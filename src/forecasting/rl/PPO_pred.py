from __future__ import annotations

"""
PPO_pred — PPO model selector optimised for **prediction accuracy**.

Reward : -|y_pred - y_true|  (per asset, per date)
State  : model prediction vectors + z-score normalised features
Action : choose one of N candidate forecast models
Method : PPO (on-policy, Actor-Critic, clipped surrogate, GAE)

Usage
-----
    selector = PPOPredSelector(hidden_dims=(128, 64), epochs=12)
    selector.fit(train_df, candidate_models)
    selected = selector.select_predictions(test_df, output_model_name="PPO_Pred")
"""

from typing import Iterable, Sequence

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.distributions import Categorical

from ._common import (
    ActorCriticNetwork,
    TrajectoryBuffer,
    compute_gae,
    prepare_pred_dataset,
    merge_selected_predictions,
    resolve_device,
    seed_everything,
)


class PPOPredSelector:
    """
    PPO agent that selects the forecast model with the lowest
    prediction error for each (date, asset) pair.
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
        self.device = resolve_device(device)
        self.verbose = verbose

        self.candidate_models_: list[str] | None = None
        self.model_fill_values_: pd.Series | None = None
        self.ac_net_: ActorCriticNetwork | None = None
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
    ) -> "PPOPredSelector":
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

        # Fallback
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
                print(f"[PPO_pred] Not enough data (n={len(states)}), "
                      f"fallback → {self.fallback_model_}")
            self.candidate_models_ = candidate_models
            return self

        n_actions = len(candidate_models)
        input_dim = states.shape[1]
        dev = torch.device(self.device)

        self.ac_net_ = ActorCriticNetwork(
            input_dim, n_actions, self.hidden_dims
        ).to(dev)
        self.optimizer_ = torch.optim.Adam(self.ac_net_.parameters(), lr=self.lr)

        all_pl: list[float] = []
        all_vl: list[float] = []
        all_ent: list[float] = []

        for ep in range(self.epochs):
            # 1. Collect trajectories
            buf = TrajectoryBuffer()
            shuffled = episodes.copy()
            self._rng.shuffle(shuffled)

            with torch.no_grad():
                for seq in shuffled:
                    for pos, idx in enumerate(seq):
                        s_np = states[idx]
                        s_t = torch.tensor(s_np, dtype=torch.float32, device=dev).unsqueeze(0)
                        logits, val = self.ac_net_(s_t)
                        dist = Categorical(logits=logits.squeeze(0))
                        action = dist.sample()
                        buf.push(
                            state=s_np,
                            action=action.item(),
                            reward=float(rewards[idx, action.item()]),
                            done=(pos == len(seq) - 1),
                            log_prob=dist.log_prob(action).item(),
                            value=val.squeeze(0).item(),
                        )

            if len(buf) == 0:
                continue

            # 2. GAE
            data = buf.to_tensors(dev)
            adv, ret = compute_gae(
                data["rewards"], data["old_values"], data["dones"],
                self.gamma, self.lam,
            )
            std = adv.std()
            if std > 1e-8:
                adv = (adv - adv.mean()) / std

            # 3. PPO updates
            n = len(buf)
            indices = np.arange(n)
            for _ in range(self.ppo_update_epochs):
                self._rng.shuffle(indices)
                for start in range(0, n, self.mini_batch_size):
                    mb = indices[start: start + self.mini_batch_size]
                    mb_s = data["states"][mb]
                    mb_a = data["actions"][mb]
                    mb_olp = data["old_log_probs"][mb]
                    mb_adv = adv[mb]
                    mb_ret = ret[mb]

                    logits, vals = self.ac_net_(mb_s)
                    dist = Categorical(logits=logits)
                    nlp = dist.log_prob(mb_a)
                    ent = dist.entropy().mean()

                    ratio = torch.exp(nlp - mb_olp)
                    s1 = ratio * mb_adv
                    s2 = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * mb_adv
                    pl = -torch.min(s1, s2).mean()
                    vl = nn.functional.mse_loss(vals, mb_ret)
                    loss = pl + self.value_coef * vl - self.entropy_coef * ent

                    self.optimizer_.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(self.ac_net_.parameters(), self.max_grad_norm)
                    self.optimizer_.step()

                    all_pl.append(pl.item())
                    all_vl.append(vl.item())
                    all_ent.append(ent.item())

            if self.verbose:
                print(
                    f"[PPO_pred] ep={ep+1}/{self.epochs} "
                    f"pl={np.mean(all_pl[-50:]):.6f} "
                    f"vl={np.mean(all_vl[-50:]):.6f} "
                    f"ent={np.mean(all_ent[-50:]):.4f}"
                )

        self.candidate_models_ = candidate_models
        return self

    # -----------------------------------------------------------------
    # select_predictions
    # -----------------------------------------------------------------
    def select_predictions(
        self,
        inference_df: pd.DataFrame,
        output_model_name: str = "PPO_Pred",
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
                columns=list(inference_df.columns) + ["selected_model", "selector_action_prob"]
            )

        if self.ac_net_ is None:
            chosen = np.full(len(states), self.fallback_model_, dtype=object)
            probs = np.full(len(states), np.nan, dtype=float)
        else:
            dev = torch.device(self.device)
            with torch.no_grad():
                logits = self.ac_net_.get_policy(
                    torch.tensor(states, dtype=torch.float32, device=dev)
                )
                dist = Categorical(logits=logits)
                idx = dist.probs.argmax(dim=1).cpu().numpy()
                probs = dist.probs[torch.arange(len(idx)), idx].cpu().numpy()
            chosen = np.array([candidate_models[i] for i in idx], dtype=object)

        return merge_selected_predictions(
            keys, chosen, {"selector_action_prob": probs},
            inference_df, output_model_name,
        )

    def _check_fitted(self) -> None:
        if self.candidate_models_ is None or self.model_fill_values_ is None:
            raise ValueError("PPOPredSelector not fitted.")


# =============================================================================
# Convenience wrapper
# =============================================================================
def build_ppo_pred_selected_predictions(
    per_prediction: pd.DataFrame,
    candidate_models: Sequence[str],
    train_split: str = "view_build",
    infer_splits: Sequence[str] = ("view_build", "test"),
    output_model_name: str = "PPO_Pred",
    params: dict | None = None,
    verbose: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Train PPO_pred per fold, emit selected predictions."""
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

        sel = PPOPredSelector(**{**params, "verbose": verbose})
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
            "used_ppo": int(sel.ac_net_ is not None),
        })

    selected = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    return selected, pd.DataFrame(diags)
