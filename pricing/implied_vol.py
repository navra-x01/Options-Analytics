"""
Implied Volatility Computation

This module computes implied volatility from market prices using root-finding
algorithms. Implied volatility is the volatility that, when plugged into the
Black-Scholes formula, produces the observed market price.

Mathematical Background:
Given a market price P_market, we solve for σ such that:
    BS(S, K, T, r, σ) = P_market

This is done using numerical root-finding methods (Brent's method or bisection).
"""

import numpy as np
import pandas as pd
from scipy.optimize import brentq, bisect
from typing import Union, Literal

from .black_scholes import black_scholes_call, black_scholes_put


def compute_implied_vol(
    market_price: float,
    S: float,
    K: float,
    T: float,
    r: float,
    option_type: str = "call",
    method: str = "brent",
    vol_min: float = 0.001,
    vol_max: float = 5.0,
) -> float:
    """
    Compute implied volatility from market price.

    Parameters
    ----------
    market_price : float
        Observed market price of the option
    S : float
        Current spot price
    K : float
        Strike price
    T : float
        Time to maturity (in years)
    r : float
        Risk-free interest rate
    option_type : str
        "call" or "put"
    method : str
        Root-finding method: "brent" (default) or "bisect"
    vol_min : float
        Minimum volatility bound (default: 0.001 = 0.1%)
    vol_max : float
        Maximum volatility bound (default: 5.0 = 500%)

    Returns
    -------
    float
        Implied volatility (annualized)

    Raises
    ------
    ValueError
        If no root is found in the given volatility range
    """
    if T <= 0:
        # At expiration, IV is undefined (price = intrinsic value)
        raise ValueError("Cannot compute IV for expired option (T <= 0)")

    if market_price <= 0:
        raise ValueError("Market price must be positive")

    option_type = option_type.lower()
    if option_type == "call":
        pricing_func = black_scholes_call
        intrinsic = max(S - K, 0.0)
    elif option_type == "put":
        pricing_func = black_scholes_put
        intrinsic = max(K - S, 0.0)
    else:
        raise ValueError(f"Invalid option_type: {option_type}. Must be 'call' or 'put'")

    # Check if market price is below intrinsic value (arbitrage)
    discounted_intrinsic = intrinsic * np.exp(-r * T)
    if market_price < discounted_intrinsic:
        raise ValueError(
            f"Market price {market_price:.4f} is below discounted intrinsic value "
            f"{discounted_intrinsic:.4f} (arbitrage opportunity)"
        )

    # Define objective function: BS_price(σ) - market_price
    def objective(sigma: float) -> float:
        try:
            bs_price = pricing_func(S, K, T, r, sigma)
            return bs_price - market_price
        except:
            return np.nan

    # Check bounds
    price_at_min = pricing_func(S, K, T, r, vol_min)
    price_at_max = pricing_func(S, K, T, r, vol_max)

    if market_price < price_at_min:
        raise ValueError(
            f"Market price {market_price:.4f} is below minimum possible price "
            f"{price_at_min:.4f} at vol={vol_min}"
        )

    if market_price > price_at_max:
        raise ValueError(
            f"Market price {market_price:.4f} is above maximum possible price "
            f"{price_at_max:.4f} at vol={vol_max}"
        )

    # Find root
    try:
        if method.lower() == "brent":
            iv = brentq(objective, vol_min, vol_max, xtol=1e-8, maxiter=100)
        elif method.lower() == "bisect":
            iv = bisect(objective, vol_min, vol_max, xtol=1e-8, maxiter=100)
        else:
            raise ValueError(f"Unknown method: {method}. Use 'brent' or 'bisect'")
    except ValueError as e:
        raise ValueError(
            f"Failed to find implied volatility: {e}. "
            f"Market price may be outside valid range."
        )

    return iv


def compute_iv_chain(
    option_chain_df: pd.DataFrame,
    S: float,
    r: float,
    strike_col: str = "strike",
    maturity_col: str = "maturity",
    option_type_col: str = "option_type",
    price_col: str = "market_price",
    method: str = "brent",
) -> pd.DataFrame:
    """
    Compute implied volatility for an entire option chain.

    Parameters
    ----------
    option_chain_df : pd.DataFrame
        DataFrame containing option chain data with columns:
        - strike (or specified strike_col)
        - maturity (or specified maturity_col) - should be in years or datetime
        - option_type (or specified option_type_col) - "call" or "put"
        - market_price (or specified price_col)
    S : float
        Current spot price
    r : float
        Risk-free interest rate
    strike_col : str
        Name of strike column (default: "strike")
    maturity_col : str
        Name of maturity column (default: "maturity")
    option_type_col : str
        Name of option type column (default: "option_type")
    price_col : str
        Name of market price column (default: "market_price")
    method : str
        Root-finding method: "brent" (default) or "bisect"

    Returns
    -------
    pd.DataFrame
        Original DataFrame with added 'implied_vol' column.
        Rows with invalid prices will have NaN in implied_vol.
    """
    df = option_chain_df.copy()

    # Convert maturity to years if it's datetime
    if pd.api.types.is_datetime64_any_dtype(df[maturity_col]):
        df["T"] = (df[maturity_col] - pd.Timestamp.now()).dt.days / 365.25
    elif isinstance(df[maturity_col].iloc[0], str):
        # Try to parse as datetime
        try:
            df["T"] = (
                pd.to_datetime(df[maturity_col]) - pd.Timestamp.now()
            ).dt.days / 365.25
        except:
            # Assume already in years
            df["T"] = df[maturity_col]
    else:
        df["T"] = df[maturity_col]

    # Compute IV for each row
    iv_list = []
    for idx, row in df.iterrows():
        try:
            # Skip if price is missing or invalid
            if pd.isna(row[price_col]) or row[price_col] <= 0:
                iv_list.append(np.nan)
                continue

            # Skip if maturity is invalid
            if pd.isna(row["T"]) or row["T"] <= 0:
                iv_list.append(np.nan)
                continue

            iv = compute_implied_vol(
                market_price=row[price_col],
                S=S,
                K=row[strike_col],
                T=row["T"],
                r=r,
                option_type=row[option_type_col],
                method=method,
            )
            iv_list.append(iv)
        except (ValueError, KeyError) as e:
            # If IV computation fails, store NaN
            iv_list.append(np.nan)

    df["implied_vol"] = iv_list
    return df
