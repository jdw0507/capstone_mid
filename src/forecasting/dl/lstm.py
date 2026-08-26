from __future__ import annotations

from typing import Optional

import torch
from torch import nn

from src.forecasting.dl.base_torch import BaseTorchSequenceForecaster


class _LSTMRegressor(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        num_layers: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        lstm_dropout = dropout if num_layers > 1 else 0.0

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=lstm_dropout,
            batch_first=True,
        )
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, F]
        out, _ = self.lstm(x)
        h = out[:, -1, :]
        y = self.head(h)
        return y


class LSTMForecaster(BaseTorchSequenceForecaster):
    """
    LSTM forecaster

    사용 예:
    model = LSTMForecaster(
        lookback=20,
        hidden_dim=64,
        num_layers=2,
        epochs=20,
        verbose=True,
    )
    """

    def __init__(
        self,
        hidden_dim: int = 64,
        num_layers: int = 2,
        lookback: int = 20,
        batch_size: int = 256,
        epochs: int = 20,
        lr: float = 1e-3,
        weight_decay: float = 1e-5,
        dropout: float = 0.1,
        val_split: float = 0.1,
        early_stopping_patience: int = 5,
        gradient_clip_norm: float | None = 1.0,
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
            device=device,
            verbose=verbose,
            log_every_n_epochs=log_every_n_epochs,
            random_state=random_state,
            model_name=model_name or "LSTM",
            feature_columns=feature_columns,
            target_column=target_column,
            horizon_days=horizon_days,
            periods_per_year=periods_per_year,
        )

        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

    def _build_network(self, input_dim: int) -> nn.Module:
        return _LSTMRegressor(
            input_dim=input_dim,
            hidden_dim=self.hidden_dim,
            num_layers=self.num_layers,
            dropout=self.dropout,
        )