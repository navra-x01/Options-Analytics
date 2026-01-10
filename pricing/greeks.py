"""
Greeks Calculation Module

This module computes the option Greeks (sensitivities) analytically and using
finite-difference methods.

Mathematical Background:
- Delta (Δ): Rate of change of option price with respect to spot price
    Call: Δ = N(d1)
    Put:  Δ = N(d1) - 1

- Gamma (Γ): Rate of change of delta with respect to spot price
    Γ = n(d1) / (S*σ*√T)
    where n(·) is the PDF of standard normal distribution

- Theta (Θ): Rate of change of option price with respect to time
    Call: Θ = -S*n(d1)*σ/(2*√T) - r*K*e^(-rT)*N(d2)
    Put:  Θ = -S*n(d1)*σ/(2*√T) + r*K*e^(-rT)*N(-d2)

- Vega (ν): Rate of change of option price with respect to volatility
    ν = S*n(d1)*√T

All Greeks are computed analytically and can also be approximated using
finite-difference methods for validation.
"""

import numpy as np
from scipy.stats import norm
from typing import Dict, Literal

from .black_scholes import black_scholes_call, black_scholes_put


def _compute_d1_d2(S: float, K: float, T: float, r: float, sigma: float) -> tuple:
    """Helper function to compute d1 and d2."""
    sqrt_T = np.sqrt(T)
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * sqrt_T)
    d2 = d1 - sigma * sqrt_T
    return d1, d2


def delta_call(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """
    Calculate the delta of a European call option.

    Delta measures the rate of change of option price with respect to spot price.
    For a call: Δ = N(d1)

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

    Returns
    -------
    float
        Delta of the call option (between 0 and 1)
    """
    if T <= 0:
        return 1.0 if S > K else 0.0

    if sigma <= 0:
        return 1.0 if S > K else 0.0

    d1, _ = _compute_d1_d2(S, K, T, r, sigma)
    return norm.cdf(d1)


def delta_put(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """
    Calculate the delta of a European put option.

    Delta measures the rate of change of option price with respect to spot price.
    For a put: Δ = N(d1) - 1

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

    Returns
    -------
    float
        Delta of the put option (between -1 and 0)
    """
    if T <= 0:
        return -1.0 if S < K else 0.0

    if sigma <= 0:
        return -1.0 if S < K else 0.0

    d1, _ = _compute_d1_d2(S, K, T, r, sigma)
    return norm.cdf(d1) - 1.0


def gamma(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """
    Calculate the gamma of a European option (same for call and put).

    Gamma measures the rate of change of delta with respect to spot price.
    Γ = n(d1) / (S*σ*√T)

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

    Returns
    -------
    float
        Gamma of the option (always positive)
    """
    if T <= 0 or sigma <= 0:
        return 0.0

    d1, _ = _compute_d1_d2(S, K, T, r, sigma)
    n_d1 = norm.pdf(d1)  # PDF of standard normal
    sqrt_T = np.sqrt(T)
    return n_d1 / (S * sigma * sqrt_T)


def theta_call(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """
    Calculate the theta of a European call option (annualized).

    Theta measures the rate of change of option price with respect to time.
    For a call: Θ = -S*n(d1)*σ/(2*√T) - r*K*e^(-rT)*N(d2)

    Note: Theta is typically negative (time decay).

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

    Returns
    -------
    float
        Theta of the call option (annualized, typically negative)
    """
    if T <= 0:
        return 0.0

    if sigma <= 0:
        return -r * K * np.exp(-r * T) if S > K else 0.0

    d1, d2 = _compute_d1_d2(S, K, T, r, sigma)
    n_d1 = norm.pdf(d1)
    sqrt_T = np.sqrt(T)

    theta = -S * n_d1 * sigma / (2 * sqrt_T) - r * K * np.exp(-r * T) * norm.cdf(d2)
    return theta


def theta_put(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """
    Calculate the theta of a European put option (annualized).

    Theta measures the rate of change of option price with respect to time.
    For a put: Θ = -S*n(d1)*σ/(2*√T) + r*K*e^(-rT)*N(-d2)

    Note: Theta is typically negative (time decay).

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

    Returns
    -------
    float
        Theta of the put option (annualized, typically negative)
    """
    if T <= 0:
        return 0.0

    if sigma <= 0:
        return r * K * np.exp(-r * T) if S < K else 0.0

    d1, d2 = _compute_d1_d2(S, K, T, r, sigma)
    n_d1 = norm.pdf(d1)
    sqrt_T = np.sqrt(T)

    theta = -S * n_d1 * sigma / (2 * sqrt_T) + r * K * np.exp(-r * T) * norm.cdf(-d2)
    return theta


def vega(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """
    Calculate the vega of a European option (same for call and put).

    Vega measures the rate of change of option price with respect to volatility.
    ν = S*n(d1)*√T

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

    Returns
    -------
    float
        Vega of the option (always positive)
    """
    if T <= 0 or sigma <= 0:
        return 0.0

    d1, _ = _compute_d1_d2(S, K, T, r, sigma)
    n_d1 = norm.pdf(d1)
    sqrt_T = np.sqrt(T)
    return S * n_d1 * sqrt_T


def compute_all_greeks(
    S: float, K: float, T: float, r: float, sigma: float, option_type: str = "call"
) -> Dict[str, float]:
    """
    Compute all Greeks for a European option.

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

    Returns
    -------
    dict
        Dictionary containing all Greeks: delta, gamma, theta, vega
    """
    option_type = option_type.lower()
    if option_type == "call":
        delta = delta_call(S, K, T, r, sigma)
        theta = theta_call(S, K, T, r, sigma)
    elif option_type == "put":
        delta = delta_put(S, K, T, r, sigma)
        theta = theta_put(S, K, T, r, sigma)
    else:
        raise ValueError(f"Invalid option_type: {option_type}. Must be 'call' or 'put'")

    return {
        "delta": delta,
        "gamma": gamma(S, K, T, r, sigma),
        "theta": theta,
        "vega": vega(S, K, T, r, sigma),
    }


def compute_greeks_fd(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    option_type: str = "call",
    bump: float = 0.01,
) -> Dict[str, float]:
    """
    Compute Greeks using finite-difference approximation.

    This method is useful for validation and for cases where analytic
    formulas are not available.

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
    bump : float
        Bump size for finite differences (default: 0.01 = 1%)

    Returns
    -------
    dict
        Dictionary containing Greeks computed via finite differences
    """
    option_type = option_type.lower()
    pricing_func = black_scholes_call if option_type == "call" else black_scholes_put

    # Current price
    price_0 = pricing_func(S, K, T, r, sigma)

    # Delta: dP/dS
    price_up = pricing_func(S * (1 + bump), K, T, r, sigma)
    price_down = pricing_func(S * (1 - bump), K, T, r, sigma)
    delta_fd = (price_up - price_down) / (2 * S * bump)

    # Gamma: d²P/dS²
    gamma_fd = (price_up - 2 * price_0 + price_down) / (S * bump) ** 2

    # Theta: -dP/dT (negative because time decreases)
    if T > bump / 365:  # Avoid negative T
        price_T_down = pricing_func(S, K, T - bump / 365, r, sigma)
        theta_fd = -(price_T_down - price_0) / (bump / 365)
    else:
        theta_fd = 0.0

    # Vega: dP/dσ
    price_sigma_up = pricing_func(S, K, T, r, sigma * (1 + bump))
    price_sigma_down = pricing_func(S, K, T, r, sigma * (1 - bump))
    vega_fd = (price_sigma_up - price_sigma_down) / (2 * sigma * bump)

    return {
        "delta": delta_fd,
        "gamma": gamma_fd,
        "theta": theta_fd,
        "vega": vega_fd,
    }
