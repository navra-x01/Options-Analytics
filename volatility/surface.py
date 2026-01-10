"""
Volatility Surface Analysis

This module builds and visualizes volatility surfaces from option chain data.
A volatility surface shows how implied volatility varies across both strike
prices and maturities.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm
import plotly.graph_objects as go
from typing import Union, Optional

from pricing.implied_vol import compute_iv_chain


def build_volatility_surface(
    option_chain_df: pd.DataFrame,
    S: float,
    r: float,
    strike_col: str = "strike",
    maturity_col: str = "maturity",
    option_type_col: str = "option_type",
    price_col: str = "market_price",
    option_type_filter: Optional[str] = None,
) -> tuple:
    """
    Build volatility surface from option chain.

    Parameters
    ----------
    option_chain_df : pd.DataFrame
        Option chain DataFrame
    S : float
        Current spot price
    r : float
        Risk-free interest rate
    strike_col : str
        Name of strike column
    maturity_col : str
        Name of maturity column
    option_type_col : str
        Name of option type column
    price_col : str
        Name of market price column
    option_type_filter : str, optional
        Filter by option type ("call" or "put"). If None, uses all options.

    Returns
    -------
    tuple
        (strikes_array, maturities_array, iv_matrix)
        - strikes_array: sorted unique strikes
        - maturities_array: sorted unique maturities (in years)
        - iv_matrix: 2D array where rows are maturities, columns are strikes
    """
    df = option_chain_df.copy()

    # Filter by option type if specified
    if option_type_filter is not None:
        df = df[df[option_type_col].str.lower() == option_type_filter.lower()].copy()

    # Convert maturity to years
    if pd.api.types.is_datetime64_any_dtype(df[maturity_col]):
        df["T"] = (df[maturity_col] - pd.Timestamp.now()).dt.days / 365.25
    elif isinstance(df[maturity_col].iloc[0], str):
        try:
            df["T"] = (
                pd.to_datetime(df[maturity_col]) - pd.Timestamp.now()
            ).dt.days / 365.25
        except:
            df["T"] = df[maturity_col]
    else:
        df["T"] = df[maturity_col]

    # Compute IVs
    df_with_iv = compute_iv_chain(
        df, S, r, strike_col, "T", option_type_col, price_col
    )

    # Get unique strikes and maturities
    strikes = sorted(df_with_iv[strike_col].unique())
    maturities = sorted(df_with_iv["T"].unique())

    # Create IV matrix
    iv_matrix = np.full((len(maturities), len(strikes)), np.nan)

    for i, T in enumerate(maturities):
        for j, K in enumerate(strikes):
            # Find matching option
            mask = (df_with_iv["T"] == T) & (df_with_iv[strike_col] == K)
            matching = df_with_iv[mask]

            if len(matching) > 0:
                # Take first valid IV (or mean if multiple)
                iv_values = matching["implied_vol"].dropna()
                if len(iv_values) > 0:
                    iv_matrix[i, j] = iv_values.iloc[0]

    return np.array(strikes), np.array(maturities), iv_matrix


def plot_volatility_surface(
    strikes: np.ndarray,
    maturities: np.ndarray,
    iv_matrix: np.ndarray,
    title: str = "Volatility Surface",
    use_plotly: bool = True,
) -> Union[go.Figure, tuple]:
    """
    Plot volatility surface in 3D and as a heatmap.

    Parameters
    ----------
    strikes : np.ndarray
        Strike prices
    maturities : np.ndarray
        Maturities (in years)
    iv_matrix : np.ndarray
        2D array of implied volatilities (rows=maturities, cols=strikes)
    title : str
        Plot title
    use_plotly : bool
        If True, return Plotly 3D surface; if False, return Matplotlib heatmap

    Returns
    -------
    plotly.graph_objects.Figure or matplotlib figure
        Plot figure
    """
    if use_plotly:
        # Create 3D surface plot
        fig = go.Figure(
            data=[
                go.Surface(
                    x=strikes,
                    y=maturities,
                    z=iv_matrix,
                    colorscale="Viridis",
                    colorbar=dict(title="Implied Volatility"),
                )
            ]
        )

        fig.update_layout(
            title=title,
            scene=dict(
                xaxis_title="Strike Price",
                yaxis_title="Time to Maturity (years)",
                zaxis_title="Implied Volatility",
            ),
            template="plotly_white",
        )

        return fig
    else:
        # Create heatmap
        fig, ax = plt.subplots(figsize=(12, 8))

        # Mask NaN values for better visualization
        masked_matrix = np.ma.masked_invalid(iv_matrix)

        im = ax.imshow(
            masked_matrix,
            aspect="auto",
            origin="lower",
            cmap=cm.viridis,
            interpolation="nearest",
        )

        # Set ticks
        n_strikes = len(strikes)
        n_maturities = len(maturities)

        # Show subset of strikes/maturities to avoid crowding
        strike_step = max(1, n_strikes // 10)
        maturity_step = max(1, n_maturities // 10)

        ax.set_xticks(np.arange(0, n_strikes, strike_step))
        ax.set_xticklabels([f"{strikes[i]:.0f}" for i in range(0, n_strikes, strike_step)])

        ax.set_yticks(np.arange(0, n_maturities, maturity_step))
        ax.set_yticklabels(
            [f"{maturities[i]:.3f}" for i in range(0, n_maturities, maturity_step)]
        )

        ax.set_xlabel("Strike Price", fontsize=12)
        ax.set_ylabel("Time to Maturity (years)", fontsize=12)
        ax.set_title(title, fontsize=14, fontweight="bold")

        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label("Implied Volatility", fontsize=12)

        plt.tight_layout()
        return fig
