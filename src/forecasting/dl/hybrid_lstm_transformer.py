from __future__ import annotations

from typing import Optional

import torch
from torch import nn

from src.forecasting.dl.base_torch import BaseTorchSequenceForecaster


class _HybridLSTMTransformerRegressor(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        lstm_layers: int = 1,
        d_model: int = 64,
        n_heads: int = 4,
        transformer_layers: int = 1,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=lstm_layers,
            dropout=dropout if lstm_layers > 1 else 0.0,
            batch_first=True,
        )

        self.proj = nn.Linear(hidden_dim, d_model)
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
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=transformer_layers)

        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, F]
        h_lstm, _ = self.lstm(x)         # [B, T, hidden_dim]
        h = self.proj(h_lstm)            # [B, T, d_model]
        h = self.norm(h)
        h = self.dropout(h)
        h = self.encoder(h)
        h = h.mean(dim=1)
        return self.head(h)


class HybridLSTMTransformerForecaster(BaseTorchSequenceForecaster):
    """
    Basic hybrid forecaster: LSTM + Transformer
    """

    def __init__(
        self,
        hidden_dim: int = 64,
        lstm_layers: int = 1,
        d_model: int = 64,
        n_heads: int = 4,
        transformer_layers: int = 1,
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
            model_name=model_name or "HybridLSTMTransformer",
            feature_columns=feature_columns,
            target_column=target_column,
            horizon_days=horizon_days,
            periods_per_year=periods_per_year,
        )

        self.hidden_dim = hidden_dim
        self.lstm_layers = lstm_layers
        self.d_model = d_model
        self.n_heads = n_heads
        self.transformer_layers = transformer_layers

    def _build_network(self, input_dim: int) -> nn.Module:
        return _HybridLSTMTransformerRegressor(
            input_dim=input_dim,
            hidden_dim=self.hidden_dim,
            lstm_layers=self.lstm_layers,
            d_model=self.d_model,
            n_heads=self.n_heads,
            transformer_layers=self.transformer_layers,
            dropout=self.dropout,
        )