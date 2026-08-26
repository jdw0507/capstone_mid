from __future__ import annotations

from typing import Optional

import torch
from torch import nn

from src.forecasting.dl.base_torch import BaseTorchSequenceForecaster


class _PatchTSTRegressor(nn.Module):
    def __init__(
        self,
        input_dim: int,
        lookback: int,
        patch_len: int = 12,
        stride: int = 6,
        d_model: int = 64,
        n_heads: int = 4,
        num_layers: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        self.input_dim = input_dim
        self.lookback = lookback
        self.patch_len = patch_len
        self.stride = stride
        self.d_model = d_model

        starts = list(range(0, lookback - patch_len + 1, stride))
        if len(starts) == 0:
            raise ValueError(
                f"lookback={lookback}, patch_len={patch_len}, stride={stride} 조합이 유효하지 않습니다."
            )
        self.patch_starts = starts
        self.n_patches = len(starts)

        self.patch_proj = nn.Linear(input_dim * patch_len, d_model)
        self.pos_embed = nn.Parameter(torch.zeros(1, self.n_patches, d_model))
        self.token_norm = nn.LayerNorm(d_model)
        self.dropout_layer = nn.Dropout(dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=4 * d_model,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1),
        )

        self._reset_parameters()

    def _reset_parameters(self) -> None:
        nn.init.normal_(self.pos_embed, mean=0.0, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, F]
        patches = []
        for s in self.patch_starts:
            patch = x[:, s : s + self.patch_len, :]  # [B, patch_len, F]
            patch = patch.reshape(x.size(0), self.patch_len * self.input_dim)
            patches.append(patch)

        tokens = torch.stack(patches, dim=1)         # [B, n_patches, patch_len*F]
        h = self.patch_proj(tokens)                  # [B, n_patches, d_model]
        h = self.token_norm(h + self.pos_embed)
        h = self.dropout_layer(h)
        h = self.encoder(h)
        h = h.mean(dim=1)                            # global average pooling
        y = self.head(h)                             # [B, 1]
        return y


class PatchTSTForecaster(BaseTorchSequenceForecaster):
    """
    Patch-based Transformer baseline

    디버깅 안정화 버전:
    - feature/target scaling
    - Huber loss
    - 낮은 learning rate
    - positional embedding 포함
    """

    def __init__(
        self,
        patch_len: int = 12,
        stride: int = 6,
        d_model: int = 64,
        n_heads: int = 4,
        num_layers: int = 2,
        lookback: int = 60,
        batch_size: int = 256,
        epochs: int = 20,
        lr: float = 3e-4,
        weight_decay: float = 1e-5,
        dropout: float = 0.1,
        val_split: float = 0.1,
        early_stopping_patience: int = 5,
        gradient_clip_norm: float | None = 1.0,
        feature_scaling: bool = True,
        target_scaling: bool = True,
        loss_name: str = "huber",
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
            lookback=lookback,
            batch_size=batch_size,
            epochs=epochs,
            lr=lr,
            weight_decay=weight_decay,
            dropout=dropout,
            val_split=val_split,
            early_stopping_patience=early_stopping_patience,
            gradient_clip_norm=gradient_clip_norm,
            feature_scaling=feature_scaling,
            target_scaling=target_scaling,
            loss_name=loss_name,
            huber_delta=huber_delta,
            device=device,
            verbose=verbose,
            log_every_n_epochs=log_every_n_epochs,
            random_state=random_state,
            model_name=model_name or "PatchTST",
            feature_columns=feature_columns,
            target_column=target_column,
            horizon_days=horizon_days,
            periods_per_year=periods_per_year,
        )

        self.patch_len = patch_len
        self.stride = stride
        self.d_model = d_model
        self.n_heads = n_heads
        self.num_layers = num_layers

    def _build_network(self, input_dim: int) -> nn.Module:
        return _PatchTSTRegressor(
            input_dim=input_dim,
            lookback=self.lookback,
            patch_len=self.patch_len,
            stride=self.stride,
            d_model=self.d_model,
            n_heads=self.n_heads,
            num_layers=self.num_layers,
            dropout=self.dropout,
        )