from __future__ import annotations

from typing import Optional

import torch
from torch import nn

from src.forecasting.dl.base_torch import BaseTorchSequenceForecaster


class _MLPRegressor(nn.Module):
    def __init__(
        self,
        input_dim: int,
        lookback: int,
        hidden_dims: tuple[int, ...] = (256, 128),
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        in_dim = input_dim * lookback
        layers = []

        prev_dim = in_dim
        for h in hidden_dims:
            layers.extend(
                [
                    nn.Linear(prev_dim, h),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                ]
            )
            prev_dim = h

        layers.append(nn.Linear(prev_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, F]
        x = x.reshape(x.size(0), -1)
        return self.net(x)


class MLPForecaster(BaseTorchSequenceForecaster):
    """
    Basic MLP forecaster
    sequence를 flatten해서 사용하는 가장 단순한 딥러닝 baseline
    """

    def __init__(
        self,
        hidden_dims: tuple[int, ...] = (256, 128),
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
            model_name=model_name or "MLP",
            feature_columns=feature_columns,
            target_column=target_column,
            horizon_days=horizon_days,
            periods_per_year=periods_per_year,
        )
        self.hidden_dims = hidden_dims

    def _build_network(self, input_dim: int) -> nn.Module:
        return _MLPRegressor(
            input_dim=input_dim,
            lookback=self.lookback,
            hidden_dims=self.hidden_dims,
            dropout=self.dropout,
        )