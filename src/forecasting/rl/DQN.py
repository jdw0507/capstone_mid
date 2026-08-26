from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
import torch
from torch import nn


@dataclass
class DQNTrainingDiagnostics:
    fold_id: int
    n_samples: int
    n_assets: int
    n_models: int
    episodes: int
    steps: int
    final_epsilon: float
    avg_loss: float
    fallback_model: str | None


class _QNetwork(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, hidden_dims: Sequence[int]) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        prev = input_dim
        for h in hidden_dims:
            layers.extend([nn.Linear(prev, h), nn.ReLU()])
            prev = h
        layers.append(nn.Linear(prev, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DQNModelSelector:
    """
    DQN agent for model-selection.
    State: model prediction vector at (date, asset)
    Action: choose one model index
    Reward: -abs(pred - true)
    """

    def __init__(
        self,
        hidden_dims: Sequence[int] = (128, 64),
        gamma: float = 0.95,
        lr: float = 1e-3,
        batch_size: int = 128,
        replay_capacity: int = 50000,
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
        self.verbose = verbose

        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        self.candidate_models_: list[str] | None = None
        self.model_fill_values_: pd.Series | None = None
        self.q_net_: _QNetwork | None = None
        self.target_net_: _QNetwork | None = None
        self.optimizer_: torch.optim.Optimizer | None = None
        self.fallback_model_: str | None = None

        self._rng = np.random.default_rng(self.random_state)
        torch.manual_seed(self.random_state)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.random_state)

    def fit(
        self,
        train_df: pd.DataFrame,
        candidate_models: Iterable[str],
    ) -> "DQNModelSelector":
        candidate_models = list(candidate_models)
        if len(candidate_models) < 2:
            raise ValueError("candidate_models must have at least two models for DQN selection.")

        bundle = self._prepare_dataset(train_df, candidate_models, fit_mode=True)
        states = bundle["states"]
        rewards = bundle["rewards"]
        episodes = bundle["episodes"]

        # fallback to static best model if too small
        model_mae = (
            train_df[train_df["model"].isin(candidate_models)]
            .assign(abs_error=lambda x: (x["y_pred"] - x["y_true"]).abs())
            .groupby("model")["abs_error"]
            .mean()
            .sort_values()
        )
        self.fallback_model_ = str(model_mae.index[0]) if not model_mae.empty else candidate_models[0]

        if len(states) < 64 or len(episodes) == 0:
            if self.verbose:
                print(
                    f"[DQN] not enough samples (n={len(states)}). "
                    f"fallback to static model: {self.fallback_model_}"
                )
            self.candidate_models_ = candidate_models
            return self

        input_dim = states.shape[1]
        n_actions = len(candidate_models)

        self.q_net_ = _QNetwork(input_dim, n_actions, self.hidden_dims).to(self.device)
        self.target_net_ = _QNetwork(input_dim, n_actions, self.hidden_dims).to(self.device)
        self.target_net_.load_state_dict(self.q_net_.state_dict())
        self.target_net_.eval()

        self.optimizer_ = torch.optim.Adam(self.q_net_.parameters(), lr=self.lr)
        replay = deque(maxlen=self.replay_capacity)

        epsilon = self.epsilon_start
        global_step = 0
        losses: list[float] = []

        for ep in range(self.episodes):
            shuffled = episodes.copy()
            self._rng.shuffle(shuffled)

            for seq in shuffled:
                for pos, idx in enumerate(seq):
                    state = states[idx]

                    if self._rng.random() < epsilon:
                        action = int(self._rng.integers(0, n_actions))
                    else:
                        with torch.no_grad():
                            q = self.q_net_(torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0))
                            action = int(torch.argmax(q, dim=1).item())

                    reward = float(rewards[idx, action])

                    done = pos == (len(seq) - 1)
                    if done:
                        next_state = np.zeros_like(state, dtype=np.float32)
                    else:
                        next_state = states[seq[pos + 1]]

                    replay.append((state, action, reward, next_state, float(done)))

                    if len(replay) >= self.min_replay_size:
                        loss_val = self._train_batch(replay)
                        if np.isfinite(loss_val):
                            losses.append(loss_val)

                    global_step += 1
                    if global_step % self.target_update_interval == 0:
                        self.target_net_.load_state_dict(self.q_net_.state_dict())

            epsilon = max(self.epsilon_end, epsilon * self.epsilon_decay)

            if self.verbose:
                avg_loss = float(np.mean(losses[-200:])) if losses else np.nan
                print(
                    f"[DQN] episode={ep + 1}/{self.episodes} "
                    f"epsilon={epsilon:.4f} avg_loss={avg_loss:.6f}"
                )

        self.candidate_models_ = candidate_models
        return self

    def select_predictions(
        self,
        inference_df: pd.DataFrame,
        output_model_name: str = "DQN_Selector",
    ) -> pd.DataFrame:
        self._check_fitted()
        candidate_models = list(self.candidate_models_)

        bundle = self._prepare_dataset(inference_df, candidate_models, fit_mode=False)
        states = bundle["states"]
        keys = bundle["keys"].copy()

        if len(states) == 0:
            return pd.DataFrame(columns=list(inference_df.columns) + ["selected_model", "selector_q_value"])

        if self.q_net_ is None:
            chosen = np.full(len(states), self.fallback_model_, dtype=object)
            qvals = np.full(len(states), np.nan, dtype=float)
        else:
            with torch.no_grad():
                q = self.q_net_(torch.tensor(states, dtype=torch.float32, device=self.device)).cpu().numpy()
            action_idx = np.argmax(q, axis=1)
            chosen = np.asarray([candidate_models[i] for i in action_idx], dtype=object)
            qvals = q[np.arange(len(q)), action_idx]

        keys["selected_model"] = chosen
        keys["selector_q_value"] = qvals

        merged = keys.merge(
            inference_df,
            left_on=["date", "asset", "selected_model"],
            right_on=["date", "asset", "model"],
            how="left",
            suffixes=("", "_raw"),
        )

        merged["model"] = output_model_name
        merged["selected_model"] = merged["selected_model"].astype(str)

        # keep original schema first, then selector metadata
        base_cols = list(inference_df.columns)
        extra_cols = [c for c in ["selected_model", "selector_q_value"] if c in merged.columns]
        return merged[base_cols + extra_cols].copy()

    def _train_batch(self, replay: deque) -> float:
        assert self.q_net_ is not None and self.target_net_ is not None and self.optimizer_ is not None

        idx = self._rng.choice(len(replay), size=min(self.batch_size, len(replay)), replace=False)
        batch = [replay[i] for i in idx]

        states = torch.tensor(np.stack([b[0] for b in batch]), dtype=torch.float32, device=self.device)
        actions = torch.tensor([b[1] for b in batch], dtype=torch.int64, device=self.device).unsqueeze(1)
        rewards = torch.tensor([b[2] for b in batch], dtype=torch.float32, device=self.device).unsqueeze(1)
        next_states = torch.tensor(np.stack([b[3] for b in batch]), dtype=torch.float32, device=self.device)
        dones = torch.tensor([b[4] for b in batch], dtype=torch.float32, device=self.device).unsqueeze(1)

        q_pred = self.q_net_(states).gather(1, actions)

        with torch.no_grad():
            q_next = self.target_net_(next_states).max(dim=1, keepdim=True).values
            q_target = rewards + (1.0 - dones) * self.gamma * q_next

        loss = nn.functional.smooth_l1_loss(q_pred, q_target)
        self.optimizer_.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.q_net_.parameters(), max_norm=5.0)
        self.optimizer_.step()
        return float(loss.item())

    def _prepare_dataset(
        self,
        df: pd.DataFrame,
        candidate_models: list[str],
        fit_mode: bool,
    ) -> dict[str, object]:
        req = {"date", "asset", "model", "y_pred"}
        miss = req - set(df.columns)
        if miss:
            raise ValueError(f"Missing required columns for DQN selector: {miss}")

        work = df.copy()
        work["date"] = pd.to_datetime(work["date"])
        work = work[work["model"].isin(candidate_models)].copy()

        pred_wide = (
            work.pivot_table(index=["date", "asset"], columns="model", values="y_pred", aggfunc="mean")
            .reindex(columns=candidate_models)
            .sort_index()
        )
        if pred_wide.empty:
            return {"states": np.empty((0, 0), dtype=np.float32), "rewards": np.empty((0, 0), dtype=np.float32), "episodes": [], "keys": pd.DataFrame(columns=["date", "asset"])}

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
            raise ValueError("DQNModelSelector is not fitted yet.")


def build_dqn_selected_predictions(
    per_prediction: pd.DataFrame,
    candidate_models: Sequence[str],
    train_split: str = "view_build",
    infer_splits: Sequence[str] = ("view_build", "test"),
    output_model_name: str = "DQN_Selector",
    dqn_params: dict | None = None,
    verbose: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Train DQN selector per fold on train_split and emit selected rows for infer_splits.
    """
    if "fold_id" not in per_prediction.columns or "evaluation_split" not in per_prediction.columns:
        raise ValueError("per_prediction must include fold_id and evaluation_split for fold-wise DQN fitting.")

    dqn_params = dqn_params or {}
    selected_parts: list[pd.DataFrame] = []
    diag_rows: list[dict] = []

    fold_ids = sorted(pd.to_numeric(per_prediction["fold_id"], errors="coerce").dropna().astype(int).unique())

    for fold_id in fold_ids:
        train_df = per_prediction[
            (per_prediction["fold_id"] == fold_id)
            & (per_prediction["evaluation_split"] == train_split)
            & (per_prediction["model"].isin(candidate_models))
        ].copy()

        if train_df.empty:
            continue

        selector_kwargs = dict(dqn_params)
        selector_kwargs.setdefault("verbose", verbose)
        selector = DQNModelSelector(**selector_kwargs)
        selector.fit(train_df=train_df, candidate_models=candidate_models)

        for split_name in infer_splits:
            infer_df = per_prediction[
                (per_prediction["fold_id"] == fold_id)
                & (per_prediction["evaluation_split"] == split_name)
                & (per_prediction["model"].isin(candidate_models))
            ].copy()
            if infer_df.empty:
                continue

            chosen = selector.select_predictions(infer_df, output_model_name=output_model_name)
            selected_parts.append(chosen)

        diag_rows.append(
            {
                "fold_id": fold_id,
                "n_train_rows": int(len(train_df)),
                "n_train_assets": int(train_df["asset"].nunique()) if "asset" in train_df.columns else np.nan,
                "n_candidate_models": int(len(candidate_models)),
                "fallback_model": selector.fallback_model_,
                "used_dqn": int(selector.q_net_ is not None),
            }
        )

    selected_df = pd.concat(selected_parts, axis=0, ignore_index=True) if selected_parts else pd.DataFrame()
    diagnostics_df = pd.DataFrame(diag_rows)
    return selected_df, diagnostics_df
