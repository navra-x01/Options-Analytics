"""
Black-Scholes Option Pricing Model

This module implements the Black-Scholes formula for pricing European options.

Mathematical Background:
The Black-Scholes formula for a European call option is:
    C = S*N(d1) - K*e^(-rT)*N(d2)

For a European put option:
    P = K*e^(-rT)*N(-d2) - S*N(-d1)

where:
    d1 = (ln(S/K) + (r + σ²/2)*T) / (σ*√T)
    d2 = d1 - σ*√T

    S = current spot price
    K = strike price
    T = time to maturity (in years)
    r = risk-free interest rate (annualized)
    σ = volatility (annualized)
    N(·) = cumulative distribution function of standard normal distribution
"""

import numpy as np
from scipy.stats import norm
from typing import Union


def black_scholes_call(
    S: float, K: float, T: float, r: float, sigma: float
) -> float:
    """
    Calculate the Black-Scholes price of a European call option.

    Parameters
    ----------
    S : float
        Current spot price of the underlying asset
    K : float
        Strike price
    T : float
        Time to maturity (in years)
    r : float
        Risk-free interest rate (annualized)
    sigma : float
        Volatility (annualized)

    Returns
    -------
    float
        The theoretical price of the call option

    Examples
    --------
    >>> black_scholes_call(100, 100, 1, 0.05, 0.2)
    10.450583572185565
    """
    # Handle edge cases
    if T <= 0:
        # At expiration, return intrinsic value
        return max(S - K, 0.0)

    if sigma <= 0:
        # Zero volatility: return discounted intrinsic value
        intrinsic = max(S - K, 0.0)
        return intrinsic * np.exp(-r * T)

    if S <= 0 or K <= 0:
        raise ValueError("Spot price S and strike K must be positive")

    # Calculate d1 and d2
    sqrt_T = np.sqrt(T)
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * sqrt_T)
    d2 = d1 - sigma * sqrt_T

    # Calculate option price
    price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)

    # Ensure non-negative price
    return max(price, 0.0)


def black_scholes_put(
    S: float, K: float, T: float, r: float, sigma: float
) -> float:
    """
    Calculate the Black-Scholes price of a European put option.

    Parameters
    ----------
    S : float
        Current spot price of the underlying asset
    K : float
        Strike price
    T : float
        Time to maturity (in years)
    r : float
        Risk-free interest rate (annualized)
    sigma : float
        Volatility (annualized)

    Returns
    -------
    float
        The theoretical price of the put option

    Examples
    --------
    >>> black_scholes_put(100, 100, 1, 0.05, 0.2)
    5.573526022256971
    """
    # Handle edge cases
    if T <= 0:
        # At expiration, return intrinsic value
        return max(K - S, 0.0)

    if sigma <= 0:
        # Zero volatility: return discounted intrinsic value
        intrinsic = max(K - S, 0.0)
        return intrinsic * np.exp(-r * T)

    if S <= 0 or K <= 0:
        raise ValueError("Spot price S and strike K must be positive")

    # Calculate d1 and d2
    sqrt_T = np.sqrt(T)
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * sqrt_T)
    d2 = d1 - sigma * sqrt_T

    # Calculate option price
    price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)

    # Ensure non-negative price
    return max(price, 0.0)


def black_scholes(
    S: float, K: float, T: float, r: float, sigma: float, option_type: str = "call"
) -> float:
    """
    Calculate the Black-Scholes price of a European option (call or put).

    Parameters
    ----------
    S : float
        Current spot price of the underlying asset
    K : float
        Strike price
    T : float
        Time to maturity (in years)
    r : float
        Risk-free interest rate (annualized)
    sigma : float
        Volatility (annualized)
    option_type : str, optional
        Type of option: "call" or "put" (default: "call")

    Returns
    -------
    float
        The theoretical price of the option
    """
    option_type = option_type.lower()
    if option_type == "call":
        return black_scholes_call(S, K, T, r, sigma)
    elif option_type == "put":
        return black_scholes_put(S, K, T, r, sigma)
    else:
        raise ValueError(f"Invalid option_type: {option_type}. Must be 'call' or 'put'")
