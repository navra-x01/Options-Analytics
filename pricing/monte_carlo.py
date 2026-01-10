"""
Monte Carlo Simulation for Option Pricing

This module implements Monte Carlo simulation for pricing European options
using Geometric Brownian Motion (GBM) under the risk-neutral measure.

Mathematical Background:
Under the risk-neutral measure, the stock price follows:
    S_T = S_0 * exp((r - σ²/2)*T + σ*√T*Z)

where Z ~ N(0,1) is a standard normal random variable.

The option price is then:
    Price = e^(-rT) * E[max(S_T - K, 0)]  for call
    Price = e^(-rT) * E[max(K - S_T, 0)]  for put

The standard error of the Monte Carlo estimate is:
    SE = std(payoffs) / √n_simulations
"""

import numpy as np
from typing import Tuple, Literal, Optional

from .black_scholes import black_scholes_call, black_scholes_put


def monte_carlo_european(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    option_type: str = "call",
    n_simulations: int = 100000,
    seed: Optional[int] = None,
    use_antithetic: bool = False,
) -> Tuple[float, float]:
    """
    Price a European option using Monte Carlo simulation.

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
    n_simulations : int
        Number of Monte Carlo simulations (default: 100000)
    seed : int, optional
        Random seed for reproducibility
    use_antithetic : bool
        Whether to use antithetic variates for variance reduction (default: False)

    Returns
    -------
    tuple
        (option_price, standard_error)
    """
    if T <= 0:
        # At expiration, return intrinsic value
        if option_type.lower() == "call":
            intrinsic = max(S - K, 0.0)
        else:
            intrinsic = max(K - S, 0.0)
        return intrinsic, 0.0

    if sigma <= 0:
        # Zero volatility case
        if option_type.lower() == "call":
            intrinsic = max(S - K, 0.0)
        else:
            intrinsic = max(K - S, 0.0)
        return intrinsic * np.exp(-r * T), 0.0

    # Set random seed
    if seed is not None:
        np.random.seed(seed)

    option_type = option_type.lower()
    if option_type not in ["call", "put"]:
        raise ValueError(f"Invalid option_type: {option_type}. Must be 'call' or 'put'")

    # Generate random numbers
    if use_antithetic:
        # Use antithetic variates: for each Z, also use -Z
        n_half = n_simulations // 2
        Z = np.random.standard_normal(n_half)
        Z = np.concatenate([Z, -Z])  # Antithetic pairs
    else:
        Z = np.random.standard_normal(n_simulations)

    # Simulate stock prices at expiration using GBM
    # S_T = S_0 * exp((r - σ²/2)*T + σ*√T*Z)
    drift = (r - 0.5 * sigma ** 2) * T
    diffusion = sigma * np.sqrt(T) * Z
    S_T = S * np.exp(drift + diffusion)

    # Compute payoffs
    if option_type == "call":
        payoffs = np.maximum(S_T - K, 0.0)
    else:  # put
        payoffs = np.maximum(K - S_T, 0.0)

    # Discount to present value
    discounted_payoffs = payoffs * np.exp(-r * T)

    # Compute mean (option price) and standard error
    option_price = np.mean(discounted_payoffs)
    std_error = np.std(discounted_payoffs) / np.sqrt(n_simulations)

    return option_price, std_error


def compare_mc_bs(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    option_type: str = "call",
    n_simulations: int = 100000,
    seed: Optional[int] = None,
) -> dict:
    """
    Compare Monte Carlo price with Black-Scholes price.

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
    n_simulations : int
        Number of Monte Carlo simulations
    seed : int, optional
        Random seed for reproducibility

    Returns
    -------
    dict
        Dictionary containing:
        - mc_price: Monte Carlo price
        - bs_price: Black-Scholes price
        - error: Absolute difference
        - relative_error: Relative error (percentage)
        - std_error: Standard error of MC estimate
    """
    option_type = option_type.lower()

    # Monte Carlo price
    mc_price, std_error = monte_carlo_european(
        S, K, T, r, sigma, option_type, n_simulations, seed
    )

    # Black-Scholes price
    if option_type == "call":
        bs_price = black_scholes_call(S, K, T, r, sigma)
    else:
        bs_price = black_scholes_put(S, K, T, r, sigma)

    # Compute errors
    error = abs(mc_price - bs_price)
    relative_error = (error / bs_price * 100) if bs_price > 0 else 0.0

    return {
        "mc_price": mc_price,
        "bs_price": bs_price,
        "error": error,
        "relative_error": relative_error,
        "std_error": std_error,
    }
