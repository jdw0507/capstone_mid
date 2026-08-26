from __future__ import annotations

from typing import Optional

import torch
from torch import nn

from src.forecasting.dl.base_torch import BaseTorchSequenceForecaster


class _TransformerRegressor(nn.Module):
    def __init__(
        self,
        input_dim: int,
        lookback: int,
        d_model: int = 64,
        n_heads: int = 4,
        num_layers: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_embed = nn.Parameter(torch.zeros(1, lookback, d_model))
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

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
        h = self.input_proj(x)
        h = self.norm(h + self.pos_embed[:, : h.size(1), :])
        h = self.dropout(h)
        h = self.encoder(h)
        h = h.mean(dim=1)
        return self.head(h)


class TransformerForecaster(BaseTorchSequenceForecaster):
    """
    Basic Transformer forecaster
    """

    def __init__(
        self,
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
            model_name=model_name or "Transformer",
            feature_columns=feature_columns,
            target_column=target_column,
            horizon_days=horizon_days,
            periods_per_year=periods_per_year,
        )
        self.d_model = d_model
        self.n_heads = n_heads
        self.num_layers = num_layers

    def _build_network(self, input_dim: int) -> nn.Module:
        return _TransformerRegressor(
            input_dim=input_dim,
            lookback=self.lookback,
            d_model=self.d_model,
            n_heads=self.n_heads,
            num_layers=self.num_layers,
            dropout=self.dropout,
        )