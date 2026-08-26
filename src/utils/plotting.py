from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd

from src.frontier.efficient_frontier import portfolio_return, portfolio_volatility
import plotly.graph_objects as go

from src.frontier.efficient_frontier import portfolio_return, portfolio_volatility


def plot_efficient_frontier(
    frontier: pd.DataFrame,
    optimal_weights: pd.Series | None = None,
    expected_returns: pd.Series | None = None,
    covariance: pd.DataFrame | None = None,
    title: str = "Efficient Frontier",
):
    """
    Efficient frontier 시각화.
    optimal_weights가 주어지면 해당 포트폴리오를 점으로 표시.
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(
        frontier["volatility"],
        frontier["target_return"],
        linewidth=2,
        label="Efficient Frontier",
    )

    if optimal_weights is not None and expected_returns is not None and covariance is not None:
        port_ret = portfolio_return(optimal_weights, expected_returns)
        port_vol = portfolio_volatility(optimal_weights, covariance)

        ax.scatter(
            [port_vol],
            [port_ret],
            s=80,
            marker="o",
            label="Optimal Portfolio",
        )

    ax.set_xlabel("Volatility")
    ax.set_ylabel("Expected Return")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


def plot_weights_bar(
    weights: pd.Series,
    title: str = "Portfolio Weights",
):
    """
    포트폴리오 비중 bar plot
    """
    fig, ax = plt.subplots(figsize=(10, 5))

    weights.sort_values().plot(kind="bar", ax=ax)
    ax.set_title(title)
    ax.set_ylabel("Weight")
    ax.grid(True, axis="y", alpha=0.3)

    plt.tight_layout()
    plt.show()




def plot_efficient_frontier_interactive(
    frontier: pd.DataFrame,
    random_portfolios: pd.DataFrame | None = None,
    optimal_weights: pd.Series | None = None,
    expected_returns: pd.Series | None = None,
    covariance: pd.DataFrame | None = None,
    risk_free_rate: float = 0.0,
    title: str = "Efficient Frontier",
):
    """
    Plotly 기반 인터랙티브 efficient frontier 그래프.
    hover 시 포트폴리오 비중 확인 가능.
    """
    fig = go.Figure()

    # 랜덤 포트폴리오 점들
    if random_portfolios is not None and not random_portfolios.empty:
        weight_cols = [c for c in random_portfolios.columns if c.startswith("weight_")]

        hover_texts = []
        for _, row in random_portfolios.iterrows():
            lines = [
                f"Return: {row['expected_return']:.4f}",
                f"Volatility: {row['volatility']:.4f}",
                f"Sharpe(0rf): {row['sharpe_zero_rf']:.4f}",
                "<br><b>Weights</b>",
            ]
            for col in weight_cols:
                asset = col.replace("weight_", "")
                lines.append(f"{asset}: {row[col]:.2%}")
            hover_texts.append("<br>".join(lines))

        fig.add_trace(
            go.Scatter(
                x=random_portfolios["volatility"],
                y=random_portfolios["expected_return"],
                mode="markers",
                name="Random Portfolios",
                text=hover_texts,
                hoverinfo="text",
                marker=dict(
                    size=5,
                    opacity=0.55,
                    color=random_portfolios["sharpe_zero_rf"],
                    colorbar=dict(title="Sharpe"),
                    showscale=True,
                ),
            )
        )

    # Frontier 선
    fig.add_trace(
        go.Scatter(
            x=frontier["volatility"],
            y=frontier["target_return"],
            mode="lines",
            name="Efficient Frontier",
            line=dict(width=3),
            hovertemplate="Volatility: %{x:.4f}<br>Return: %{y:.4f}<extra></extra>",
        )
    )

    # Optimal portfolio 점
    if optimal_weights is not None and expected_returns is not None and covariance is not None:
        port_ret = portfolio_return(optimal_weights, expected_returns)
        port_vol = portfolio_volatility(optimal_weights, covariance)
        port_sharpe = (port_ret - risk_free_rate) / port_vol if port_vol > 0 else None

        lines = [
            f"Return: {port_ret:.4f}",
            f"Volatility: {port_vol:.4f}",
            f"Sharpe: {port_sharpe:.4f}" if port_sharpe is not None else "Sharpe: nan",
            "<br><b>Weights</b>",
        ]
        for asset, w in optimal_weights.items():
            lines.append(f"{asset}: {w:.2%}")

        fig.add_trace(
            go.Scatter(
                x=[port_vol],
                y=[port_ret],
                mode="markers",
                name="Optimal Portfolio",
                text=["<br>".join(lines)],
                hoverinfo="text",
                marker=dict(size=12, symbol="diamond"),
            )
        )

    fig.update_layout(
        title=title,
        xaxis_title="Volatility",
        yaxis_title="Expected Return",
        template="plotly_white",
        hovermode="closest",
    )

    fig.show()