"""
PnL Simulation and Payoff Analysis

This module simulates profit and loss (PnL) for options under different
spot price scenarios and visualizes payoff diagrams.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Tuple, Union, Optional

from pricing.black_scholes import black_scholes_call, black_scholes_put
from pricing.greeks import compute_all_greeks


def simulate_pnl(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    option_type: str = "call",
    spot_range: Optional[Tuple[float, float]] = None,
    n_points: int = 100,
) -> pd.DataFrame:
    """
    Simulate PnL for an option across different spot prices.

    Parameters
    ----------
    S : float
        Initial spot price
    K : float
        Strike price
    T : float
        Time to maturity (in years)
    r : float
        Risk-free interest rate
    sigma : float
        Volatility
    option_type : str
        "call" or "put"
    spot_range : tuple, optional
        (min_spot, max_spot) range. If None, uses [0.5*S, 1.5*S]
    n_points : int
        Number of spot price points to evaluate

    Returns
    -------
    pd.DataFrame
        DataFrame with columns: ['spot', 'option_price', 'pnl']
    """
    if spot_range is None:
        spot_min = 0.5 * S
        spot_max = 1.5 * S
    else:
        spot_min, spot_max = spot_range

    spot_prices = np.linspace(spot_min, spot_max, n_points)

    # Initial option price
    if option_type.lower() == "call":
        initial_price = black_scholes_call(S, K, T, r, sigma)
        pricing_func = black_scholes_call
    else:
        initial_price = black_scholes_put(S, K, T, r, sigma)
        pricing_func = black_scholes_put

    # Compute option prices at different spot levels
    option_prices = []
    for spot in spot_prices:
        price = pricing_func(spot, K, T, r, sigma)
        option_prices.append(price)

    option_prices = np.array(option_prices)

    # Compute PnL (assuming we bought the option)
    pnl = option_prices - initial_price

    return pd.DataFrame(
        {
            "spot": spot_prices,
            "option_price": option_prices,
            "pnl": pnl,
        }
    )


def plot_payoff_diagram(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    option_type: str = "call",
    spot_range: Optional[Tuple[float, float]] = None,
    n_points: int = 100,
    show_intrinsic: bool = True,
) -> plt.Figure:
    """
    Plot payoff diagram showing option value vs spot price.

    Parameters
    ----------
    S : float
        Current spot price
    K : float
        Strike price
    T : float
        Time to maturity (in years)
    r : float
        Risk-free interest rate
    sigma : float
        Volatility
    option_type : str
        "call" or "put"
    spot_range : tuple, optional
        (min_spot, max_spot) range
    n_points : int
        Number of points to plot
    show_intrinsic : bool
        Whether to also plot intrinsic value (expiration payoff)

    Returns
    -------
    matplotlib.figure.Figure
        Plot figure
    """
    option_type = option_type.lower()

    # Simulate PnL
    df = simulate_pnl(S, K, T, r, sigma, option_type, spot_range, n_points)

    # Create plot
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot current option value
    ax.plot(
        df["spot"],
        df["option_price"],
        linewidth=2,
        label=f"Option Value (T={T:.2f} years)",
        color="blue",
    )

    # Plot intrinsic value if requested
    if show_intrinsic:
        if option_type == "call":
            intrinsic = np.maximum(df["spot"] - K, 0)
        else:
            intrinsic = np.maximum(K - df["spot"], 0)

        ax.plot(
            df["spot"],
            intrinsic,
            "--",
            linewidth=2,
            label="Intrinsic Value (at expiration)",
            color="red",
            alpha=0.7,
        )

    # Mark current spot
    if option_type == "call":
        current_price = black_scholes_call(S, K, T, r, sigma)
    else:
        current_price = black_scholes_put(S, K, T, r, sigma)

    ax.axvline(x=S, color="green", linestyle=":", linewidth=2, label=f"Current Spot: {S:.2f}")
    ax.axhline(y=current_price, color="green", linestyle=":", linewidth=2, alpha=0.5)

    # Mark strike
    ax.axvline(x=K, color="orange", linestyle="--", linewidth=1, alpha=0.5, label=f"Strike: {K:.2f}")

    ax.set_xlabel("Spot Price", fontsize=12)
    ax.set_ylabel("Option Value", fontsize=12)
    ax.set_title(
        f"{option_type.capitalize()} Option Payoff Diagram\n"
        f"(K={K:.2f}, σ={sigma:.2%}, r={r:.2%})",
        fontsize=14,
        fontweight="bold",
    )
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")

    plt.tight_layout()
    return fig


def plot_greeks_approximation(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    option_type: str = "call",
    spot_range: Optional[Tuple[float, float]] = None,
    n_points: int = 100,
) -> plt.Figure:
    """
    Compare Greeks-based linear approximation vs true option price.

    Uses first-order Taylor expansion:
        ΔP ≈ Δ * ΔS + 0.5 * Γ * (ΔS)²

    Parameters
    ----------
    S : float
        Current spot price
    K : float
        Strike price
    T : float
        Time to maturity (in years)
    r : float
        Risk-free interest rate
    sigma : float
        Volatility
    option_type : str
        "call" or "put"
    spot_range : tuple, optional
        (min_spot, max_spot) range
    n_points : int
        Number of points to plot

    Returns
    -------
    matplotlib.figure.Figure
        Plot figure showing true price vs Greeks approximation
    """
    option_type = option_type.lower()

    # Get current Greeks
    greeks = compute_all_greeks(S, K, T, r, sigma, option_type)
    delta = greeks["delta"]
    gamma = greeks["gamma"]

    # Get current option price
    if option_type == "call":
        current_price = black_scholes_call(S, K, T, r, sigma)
        pricing_func = black_scholes_call
    else:
        current_price = black_scholes_put(S, K, T, r, sigma)
        pricing_func = black_scholes_put

    # Generate spot range
    if spot_range is None:
        spot_min = 0.5 * S
        spot_max = 1.5 * S
    else:
        spot_min, spot_max = spot_range

    spot_prices = np.linspace(spot_min, spot_max, n_points)

    # True prices
    true_prices = []
    for spot in spot_prices:
        price = pricing_func(spot, K, T, r, sigma)
        true_prices.append(price)

    true_prices = np.array(true_prices)

    # Greeks approximation: ΔP ≈ Δ * ΔS + 0.5 * Γ * (ΔS)²
    delta_S = spot_prices - S
    approx_prices = current_price + delta * delta_S + 0.5 * gamma * delta_S ** 2

    # Create plot
    fig, ax = plt.subplots(figsize=(12, 6))

    ax.plot(
        spot_prices,
        true_prices,
        linewidth=2,
        label="True Option Price",
        color="blue",
    )
    ax.plot(
        spot_prices,
        approx_prices,
        "--",
        linewidth=2,
        label="Greeks Approximation (Δ + 0.5*Γ*ΔS²)",
        color="red",
    )

    # Mark current spot
    ax.axvline(x=S, color="green", linestyle=":", linewidth=2, label=f"Current Spot: {S:.2f}")

    ax.set_xlabel("Spot Price", fontsize=12)
    ax.set_ylabel("Option Price", fontsize=12)
    ax.set_title(
        f"Greeks Approximation vs True Price\n"
        f"(Δ={delta:.4f}, Γ={gamma:.4f})",
        fontsize=14,
        fontweight="bold",
    )
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")

    plt.tight_layout()
    return fig
