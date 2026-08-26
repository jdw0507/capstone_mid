from __future__ import annotations

from abc import abstractmethod
from typing import Optional

import copy
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.forecasting.base import BaseForecaster


class BaseTorchSequenceForecaster(BaseForecaster):
    """
    시퀀스 기반 PyTorch forecaster 공통 베이스

    디버깅 강화 포인트
    -----------------
    - feature scaling / target scaling
    - NaN / Inf 자동 검사
    - epoch별 train/val loss 로그
    - 예측 NaN 발생 시 바로 에러
    """

    def __init__(
        self,
        lookback: int = 20,
        batch_size: int = 256,
        epochs: int = 20,
        lr: float = 1e-3,
        weight_decay: float = 1e-5,
        dropout: float = 0.1,
        val_split: float = 0.1,
        early_stopping_patience: int = 5,
        gradient_clip_norm: float | None = 1.0,
        feature_scaling: bool = True,
        target_scaling: bool = True,
        loss_name: str = "huber",   # "mse" or "huber"
        huber_delta: float = 1.0,
        device: Optional[str] = None,
        verbose: bool = True,
        log_every_n_epochs: int = 1,
        random_state: int = 42,
        model_name: Optional[str] = None,
        feature_columns: Optional[list[str]] = None,
        target_column: str = "target_5d_excess",
        horizon_days: int = 5,
        periods_per_year: int = 252,
    ) -> None:
        super().__init__(
            model_name=model_name or self.__class__.__name__,
            feature_columns=feature_columns,
            target_column=target_column,
            horizon_days=horizon_days,
            periods_per_year=periods_per_year,
        )

        self.lookback = lookback
        self.batch_size = batch_size
        self.epochs = epochs
        self.lr = lr
        self.weight_decay = weight_decay
        self.dropout = dropout
        self.val_split = val_split
        self.early_stopping_patience = early_stopping_patience
        self.gradient_clip_norm = gradient_clip_norm
        self.feature_scaling = feature_scaling
        self.target_scaling = target_scaling
        self.loss_name = loss_name
        self.huber_delta = huber_delta
        self.verbose = verbose
        self.log_every_n_epochs = log_every_n_epochs
        self.random_state = random_state

        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        self.fill_values_: pd.Series | None = None
        self.feature_mean_: pd.Series | None = None
        self.feature_std_: pd.Series | None = None

        self.target_mean_: float | None = None
        self.target_std_: float | None = None

        self.context_by_asset_: dict[str, np.ndarray] = {}
        self.model_: nn.Module | None = None
        self.input_dim_: int | None = None

        self.global_residual_std_: float | None = None
        self.asset_residual_std_: pd.Series | None = None

        self._set_random_seed()

    def _set_random_seed(self) -> None:
        np.random.seed(self.random_state)
        torch.manual_seed(self.random_state)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.random_state)

    @abstractmethod
    def _build_network(self, input_dim: int) -> nn.Module:
        pass

    # -----------------------------
    # Core fit / predict
    # -----------------------------
    def _fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        meta: Optional[pd.DataFrame] = None,
    ) -> None:
        if meta is None or not {"date", "asset"}.issubset(meta.columns):
            raise ValueError(f"{self.model_name}는 meta에 date, asset 컬럼이 필요합니다.")

        X_fit = X.copy()
        y_fit = y.copy()
        meta_fit = meta.copy().reset_index(drop=True)

        X_fit = self._clean_features(X_fit)
        self._fit_feature_preprocessor(X_fit)
        X_fit = self._transform_features(X_fit)

        y_fit = pd.to_numeric(y_fit, errors="coerce")
        self._fit_target_preprocessor(y_fit)
        y_fit_scaled = self._transform_target(y_fit)

        seq_X, seq_y_scaled, seq_y_original, seq_meta = self._build_train_sequences(
            X_fit, y_fit_scaled, y_fit, meta_fit
        )

        if len(seq_X) == 0:
            raise ValueError(
                f"{self.model_name}: 학습 가능한 sequence가 없습니다. "
                f"lookback을 줄이거나 데이터 길이를 확인하세요."
            )

        # finite check
        valid_mask = (
            np.isfinite(seq_X).all(axis=(1, 2))
            & np.isfinite(seq_y_scaled)
            & np.isfinite(seq_y_original)
        )
        seq_X = seq_X[valid_mask]
        seq_y_scaled = seq_y_scaled[valid_mask]
        seq_y_original = seq_y_original[valid_mask]
        seq_meta = seq_meta.loc[valid_mask].reset_index(drop=True)

        if len(seq_X) == 0:
            raise ValueError(f"{self.model_name}: finite한 학습 sequence가 없습니다.")

        self.input_dim_ = seq_X.shape[-1]
        self.model_ = self._build_network(self.input_dim_).to(self.device)

        train_loader, val_loader = self._make_dataloaders(seq_X, seq_y_scaled, seq_meta)

        optimizer = torch.optim.Adam(
            self.model_.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )

        if self.loss_name == "mse":
            criterion = nn.MSELoss()
        elif self.loss_name == "huber":
            criterion = nn.HuberLoss(delta=self.huber_delta)
        else:
            raise ValueError(f"지원하지 않는 loss_name: {self.loss_name}")

        best_state = None
        best_val_loss = np.inf
        patience_counter = 0

        if self.verbose:
            print(f"\n[{self.model_name}] Training start")
            print(
                f"device={self.device}, n_seq={len(seq_X)}, lookback={self.lookback}, "
                f"input_dim={self.input_dim_}, lr={self.lr}, loss={self.loss_name}"
            )

        for epoch in range(1, self.epochs + 1):
            self.model_.train()
            train_losses = []

            for xb, yb in train_loader:
                xb = xb.to(self.device)
                yb = yb.to(self.device)

                optimizer.zero_grad()
                pred = self.model_(xb).squeeze(-1)

                if not torch.isfinite(pred).all():
                    raise RuntimeError(f"[{self.model_name}] NaN/Inf prediction detected during training.")

                loss = criterion(pred, yb)

                if not torch.isfinite(loss):
                    raise RuntimeError(f"[{self.model_name}] NaN/Inf loss detected during training.")

                loss.backward()

                if self.gradient_clip_norm is not None:
                    torch.nn.utils.clip_grad_norm_(self.model_.parameters(), self.gradient_clip_norm)

                optimizer.step()
                train_losses.append(loss.item())

            mean_train_loss = float(np.mean(train_losses)) if len(train_losses) > 0 else np.nan
            mean_val_loss = np.nan

            if val_loader is not None:
                self.model_.eval()
                val_losses = []
                with torch.no_grad():
                    for xb, yb in val_loader:
                        xb = xb.to(self.device)
                        yb = yb.to(self.device)

                        pred = self.model_(xb).squeeze(-1)
                        if not torch.isfinite(pred).all():
                            raise RuntimeError(f"[{self.model_name}] NaN/Inf prediction detected during validation.")

                        loss = criterion(pred, yb)
                        if not torch.isfinite(loss):
                            raise RuntimeError(f"[{self.model_name}] NaN/Inf val loss detected.")

                        val_losses.append(loss.item())

                mean_val_loss = float(np.mean(val_losses)) if len(val_losses) > 0 else np.nan

                if mean_val_loss < best_val_loss:
                    best_val_loss = mean_val_loss
                    best_state = copy.deepcopy(self.model_.state_dict())
                    patience_counter = 0
                else:
                    patience_counter += 1

            if self.verbose and (epoch == 1 or epoch % self.log_every_n_epochs == 0):
                if val_loader is None:
                    print(f"[{self.model_name}] epoch={epoch:03d} train_loss={mean_train_loss:.6f}")
                else:
                    print(
                        f"[{self.model_name}] epoch={epoch:03d} "
                        f"train_loss={mean_train_loss:.6f} val_loss={mean_val_loss:.6f}"
                    )

            if val_loader is not None and patience_counter >= self.early_stopping_patience:
                if self.verbose:
                    print(f"[{self.model_name}] early stopping at epoch={epoch}")
                break

        if best_state is not None:
            self.model_.load_state_dict(best_state)

        # 학습용 context 저장 (scaled feature 기준)
        self._store_train_context(X_fit, meta_fit)

        # in-sample residual std 저장 (원래 타깃 scale로)
        train_pred_scaled = self._predict_from_sequence_matrix(seq_X)
        train_pred_original = self._inverse_transform_target(train_pred_scaled)

        resid = pd.Series(seq_y_original.flatten()) - pd.Series(train_pred_original.flatten())

        self.global_residual_std_ = float(resid.std(ddof=1)) if len(resid) > 1 else 0.0

        tmp = seq_meta.copy().reset_index(drop=True)
        tmp["resid"] = resid.reset_index(drop=True)
        self.asset_residual_std_ = tmp.groupby("asset")["resid"].std(ddof=1).astype(float)

    def _predict(
        self,
        X: pd.DataFrame,
        meta: Optional[pd.DataFrame] = None,
    ) -> pd.Series | np.ndarray:
        if self.model_ is None:
            raise ValueError(f"{self.model_name} 모델이 아직 fit되지 않았습니다.")
        if meta is None or not {"date", "asset"}.issubset(meta.columns):
            raise ValueError(f"{self.model_name}는 meta에 date, asset 컬럼이 필요합니다.")

        X_pred = X.copy()
        X_pred = self._clean_features(X_pred)
        X_pred = self._transform_features(X_pred)

        meta_pred = meta.copy().reset_index(drop=True)

        seq_X, seq_meta, row_order = self._build_prediction_sequences(X_pred, meta_pred)

        full_pred = np.full(len(X_pred), np.nan, dtype=float)

        if len(seq_X) == 0:
            if self.verbose:
                print(f"[{self.model_name}] warning: 예측 가능한 sequence가 없습니다.")
            return full_pred

        if not np.isfinite(seq_X).all():
            raise RuntimeError(f"[{self.model_name}] 예측 sequence에 NaN/Inf가 존재합니다.")

        pred_scaled = self._predict_from_sequence_matrix(seq_X)
        pred_original = self._inverse_transform_target(pred_scaled)
        pred_original = np.asarray(pred_original).reshape(-1)

        full_pred[np.asarray(row_order, dtype=int)] = pred_original
        return full_pred

    def _predict_uncertainty(
        self,
        X: pd.DataFrame,
        meta: Optional[pd.DataFrame] = None,
    ) -> pd.Series | np.ndarray | None:
        if self.global_residual_std_ is None:
            return None

        if meta is not None and "asset" in meta.columns and self.asset_residual_std_ is not None:
            out = meta["asset"].map(self.asset_residual_std_).astype(float)
            out = out.fillna(self.global_residual_std_)
            return out.values

        return np.full(len(X), self.global_residual_std_, dtype=float)

    # -----------------------------
    # Preprocessing
    # -----------------------------
    def _clean_features(self, X: pd.DataFrame) -> pd.DataFrame:
        Xc = X.copy()
        Xc = Xc.apply(pd.to_numeric, errors="coerce")
        Xc = Xc.replace([np.inf, -np.inf], np.nan)
        return Xc

    def _fit_feature_preprocessor(self, X: pd.DataFrame) -> None:
        Xf = X.copy()

        self.fill_values_ = Xf.median(numeric_only=True).reindex(Xf.columns).fillna(0.0)
        Xf = Xf.fillna(self.fill_values_)

        self.feature_mean_ = Xf.mean(axis=0)
        self.feature_std_ = Xf.std(axis=0, ddof=0).replace(0.0, 1.0).fillna(1.0)

    def _transform_features(self, X: pd.DataFrame) -> pd.DataFrame:
        if self.fill_values_ is None:
            raise ValueError("feature preprocessor가 아직 fit되지 않았습니다.")

        Xf = X.copy()
        Xf = Xf.fillna(self.fill_values_)

        if self.feature_scaling:
            if self.feature_mean_ is None or self.feature_std_ is None:
                raise ValueError("feature scaling 통계량이 없습니다.")
            Xf = (Xf - self.feature_mean_) / self.feature_std_

        Xf = Xf.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        return Xf

    def _fit_target_preprocessor(self, y: pd.Series) -> None:
        y = pd.to_numeric(y, errors="coerce")
        self.target_mean_ = float(y.mean()) if self.target_scaling else 0.0

        if self.target_scaling:
            std = float(y.std(ddof=0))
            self.target_std_ = std if np.isfinite(std) and std > 0 else 1.0
        else:
            self.target_std_ = 1.0

    def _transform_target(self, y: pd.Series) -> pd.Series:
        ys = pd.to_numeric(y, errors="coerce").copy()
        if self.target_scaling:
            if self.target_mean_ is None or self.target_std_ is None:
                raise ValueError("target preprocessor가 아직 fit되지 않았습니다.")
            ys = (ys - self.target_mean_) / self.target_std_
        return ys

    def _inverse_transform_target(self, y_scaled: np.ndarray | pd.Series) -> np.ndarray:
        arr = np.asarray(y_scaled, dtype=float)
        if self.target_scaling:
            arr = arr * float(self.target_std_) + float(self.target_mean_)
        return arr

    # -----------------------------
    # Sequence builders
    # -----------------------------
    def _build_train_sequences(
        self,
        X: pd.DataFrame,
        y_scaled: pd.Series,
        y_original: pd.Series,
        meta: pd.DataFrame,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, pd.DataFrame]:
        df = pd.concat(
            [
                meta[["date", "asset"]].reset_index(drop=True),
                X.reset_index(drop=True),
                y_scaled.rename("target_scaled").reset_index(drop=True),
                y_original.rename("target_original").reset_index(drop=True),
            ],
            axis=1,
        )

        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values(["asset", "date"]).reset_index(drop=True)

        feat_cols = list(X.columns)

        seq_X_list = []
        seq_y_scaled_list = []
        seq_y_orig_list = []
        seq_meta_rows = []

        for asset, g in df.groupby("asset"):
            g = g.sort_values("date").reset_index(drop=True)

            x_vals = g[feat_cols].to_numpy(dtype=np.float32)
            y_vals_scaled = g["target_scaled"].to_numpy(dtype=np.float32)
            y_vals_orig = g["target_original"].to_numpy(dtype=np.float32)

            if len(g) < self.lookback:
                continue

            for i in range(self.lookback - 1, len(g)):
                seq_x = x_vals[i - self.lookback + 1 : i + 1]
                seq_y_scaled = y_vals_scaled[i]
                seq_y_orig = y_vals_orig[i]

                seq_X_list.append(seq_x)
                seq_y_scaled_list.append(seq_y_scaled)
                seq_y_orig_list.append(seq_y_orig)
                seq_meta_rows.append(
                    {
                        "date": g.loc[i, "date"],
                        "asset": asset,
                    }
                )

        seq_X = np.asarray(seq_X_list, dtype=np.float32)
        seq_y_scaled = np.asarray(seq_y_scaled_list, dtype=np.float32)
        seq_y_orig = np.asarray(seq_y_orig_list, dtype=np.float32)
        seq_meta = pd.DataFrame(seq_meta_rows)

        if len(seq_meta) > 0:
            order = seq_meta.sort_values(["date", "asset"]).index.to_numpy()
            seq_X = seq_X[order]
            seq_y_scaled = seq_y_scaled[order]
            seq_y_orig = seq_y_orig[order]
            seq_meta = seq_meta.iloc[order].reset_index(drop=True)

        return seq_X, seq_y_scaled, seq_y_orig, seq_meta

    def _store_train_context(self, X_scaled: pd.DataFrame, meta: pd.DataFrame) -> None:
        df = pd.concat(
            [meta[["date", "asset"]].reset_index(drop=True), X_scaled.reset_index(drop=True)],
            axis=1,
        )
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values(["asset", "date"]).reset_index(drop=True)

        feat_cols = list(X_scaled.columns)
        self.context_by_asset_ = {}

        n_ctx = max(self.lookback - 1, 0)
        if n_ctx == 0:
            return

        for asset, g in df.groupby("asset"):
            g = g.sort_values("date").reset_index(drop=True)
            self.context_by_asset_[asset] = g[feat_cols].tail(n_ctx).to_numpy(dtype=np.float32)

    def _build_prediction_sequences(
        self,
        X_scaled: pd.DataFrame,
        meta: pd.DataFrame,
    ) -> tuple[np.ndarray, pd.DataFrame, list[int]]:
        df = pd.concat(
            [
                meta[["date", "asset"]].reset_index(drop=True),
                X_scaled.reset_index(drop=True),
            ],
            axis=1,
        )
        df["date"] = pd.to_datetime(df["date"])
        df["_row_id"] = np.arange(len(df))

        feat_cols = list(X_scaled.columns)

        seq_X_list = []
        seq_meta_rows = []
        row_order = []

        for asset, g in df.groupby("asset"):
            g = g.sort_values("date").reset_index(drop=True)

            current_x = g[feat_cols].to_numpy(dtype=np.float32)
            history_x = self.context_by_asset_.get(asset, None)

            if history_x is None:
                history_x = np.empty((0, len(feat_cols)), dtype=np.float32)

            combined_x = np.vstack([history_x, current_x])
            hist_len = len(history_x)

            for j in range(len(g)):
                end = hist_len + j
                start = end - self.lookback + 1

                if start < 0:
                    continue

                seq_x = combined_x[start : end + 1]
                if len(seq_x) != self.lookback:
                    continue

                seq_X_list.append(seq_x)
                seq_meta_rows.append(
                    {
                        "date": g.loc[j, "date"],
                        "asset": asset,
                    }
                )
                row_order.append(int(g.loc[j, "_row_id"]))

        seq_X = np.asarray(seq_X_list, dtype=np.float32)
        seq_meta = pd.DataFrame(seq_meta_rows)

        return seq_X, seq_meta, row_order

    # -----------------------------
    # Training helpers
    # -----------------------------
    def _make_dataloaders(
        self,
        seq_X: np.ndarray,
        seq_y: np.ndarray,
        seq_meta: pd.DataFrame,
    ) -> tuple[DataLoader, Optional[DataLoader]]:
        n = len(seq_X)
        val_size = int(n * self.val_split) if self.val_split > 0 else 0

        if val_size <= 0 or n - val_size <= 0:
            train_ds = TensorDataset(
                torch.tensor(seq_X, dtype=torch.float32),
                torch.tensor(seq_y, dtype=torch.float32),
            )
            train_loader = DataLoader(train_ds, batch_size=self.batch_size, shuffle=True)
            return train_loader, None

        # seq_meta는 date 정렬 상태이므로 뒤쪽을 validation으로 씀
        train_X = seq_X[:-val_size]
        train_y = seq_y[:-val_size]
        val_X = seq_X[-val_size:]
        val_y = seq_y[-val_size:]

        train_ds = TensorDataset(
            torch.tensor(train_X, dtype=torch.float32),
            torch.tensor(train_y, dtype=torch.float32),
        )
        val_ds = TensorDataset(
            torch.tensor(val_X, dtype=torch.float32),
            torch.tensor(val_y, dtype=torch.float32),
        )

        train_loader = DataLoader(train_ds, batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=self.batch_size, shuffle=False)

        return train_loader, val_loader

    def _predict_from_sequence_matrix(self, seq_X: np.ndarray) -> np.ndarray:
        if self.model_ is None:
            raise ValueError(f"{self.model_name} 모델이 아직 fit되지 않았습니다.")

        self.model_.eval()

        preds = []
        with torch.no_grad():
            for start in range(0, len(seq_X), self.batch_size):
                batch = torch.tensor(
                    seq_X[start : start + self.batch_size],
                    dtype=torch.float32,
                ).to(self.device)

                pred = self.model_(batch).squeeze(-1).detach().cpu().numpy()
                if not np.isfinite(pred).all():
                    raise RuntimeError(f"[{self.model_name}] 예측 단계에서 NaN/Inf 출력 발생.")
                preds.append(pred)

        return np.concatenate(preds, axis=0)