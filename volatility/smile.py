"""
Volatility Smile Analysis

This module builds and visualizes volatility smiles from option chain data.
A volatility smile shows how implied volatility varies with strike price
for a fixed maturity.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from typing import Union, Optional
from datetime import datetime

from pricing.implied_vol import compute_iv_chain


def build_volatility_smile(
    option_chain_df: pd.DataFrame,
    maturity_date: Union[str, datetime, float],
    S: float,
    r: float,
    strike_col: str = "strike",
    maturity_col: str = "maturity",
    option_type_col: str = "option_type",
    price_col: str = "market_price",
) -> pd.DataFrame:
    """
    Build volatility smile for a specific maturity.

    Parameters
    ----------
    option_chain_df : pd.DataFrame
        Option chain DataFrame
    maturity_date : str, datetime, or float
        Target maturity (datetime string, datetime object, or years)
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

    Returns
    -------
    pd.DataFrame
        DataFrame with columns: ['strike', 'implied_vol', 'option_type']
        Filtered to the specified maturity
    """
    df = option_chain_df.copy()

    # Convert maturity_date to comparable format
    if isinstance(maturity_date, (int, float)):
        # Assume it's already in years
        target_T = maturity_date
        if pd.api.types.is_datetime64_any_dtype(df[maturity_col]):
            df["T"] = (df[maturity_col] - pd.Timestamp.now()).dt.days / 365.25
        else:
            df["T"] = df[maturity_col]
    else:
        # Convert to datetime
        if isinstance(maturity_date, str):
            target_date = pd.to_datetime(maturity_date)
        else:
            target_date = maturity_date

        if pd.api.types.is_datetime64_any_dtype(df[maturity_col]):
            df["T"] = (df[maturity_col] - pd.Timestamp.now()).dt.days / 365.25
            target_T = (target_date - pd.Timestamp.now()).days / 365.25
        else:
            # Assume maturity_col is already in years
            df["T"] = df[maturity_col]
            target_T = (target_date - pd.Timestamp.now()).days / 365.25

    # Filter by maturity (allow small tolerance for floating point)
    tolerance = 1 / 365.25  # ~1 day tolerance
    df_filtered = df[abs(df["T"] - target_T) < tolerance].copy()

    if len(df_filtered) == 0:
        raise ValueError(
            f"No options found for maturity {maturity_date}. "
            f"Available maturities: {df['T'].unique()}"
        )

    # Compute IVs
    df_with_iv = compute_iv_chain(
        df_filtered, S, r, strike_col, "T", option_type_col, price_col
    )

    # Extract relevant columns
    result = df_with_iv[[strike_col, "implied_vol", option_type_col]].copy()
    result.columns = ["strike", "implied_vol", "option_type"]
    result = result.dropna(subset=["implied_vol"])

    # Sort by strike
    result = result.sort_values("strike")

    return result


def plot_volatility_smile(
    strikes: Union[np.ndarray, pd.Series],
    ivs: Union[np.ndarray, pd.Series],
    title: str = "Volatility Smile",
    spot_price: Optional[float] = None,
    use_plotly: bool = True,
) -> Union[go.Figure, plt.Figure]:
    """
    Plot volatility smile.

    Parameters
    ----------
    strikes : array-like
        Strike prices
    ivs : array-like
        Implied volatilities
    title : str
        Plot title
    spot_price : float, optional
        Current spot price (will be marked with vertical line)
    use_plotly : bool
        If True, return Plotly figure; if False, return Matplotlib figure

    Returns
    -------
    plotly.graph_objects.Figure or matplotlib.figure.Figure
        Plot figure
    """
    if use_plotly:
        fig = go.Figure()

        fig.add_trace(
            go.Scatter(
                x=strikes,
                y=ivs,
                mode="lines+markers",
                name="Implied Volatility",
                line=dict(width=2),
                marker=dict(size=6),
            )
        )

        if spot_price is not None:
            fig.add_vline(
                x=spot_price,
                line_dash="dash",
                line_color="red",
                annotation_text="Spot Price",
            )

        fig.update_layout(
            title=title,
            xaxis_title="Strike Price",
            yaxis_title="Implied Volatility",
            hovermode="x unified",
            template="plotly_white",
        )

        return fig
    else:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(strikes, ivs, "o-", linewidth=2, markersize=6, label="Implied Volatility")

        if spot_price is not None:
            ax.axvline(
                x=spot_price, color="red", linestyle="--", label="Spot Price", linewidth=2
            )

        ax.set_xlabel("Strike Price", fontsize=12)
        ax.set_ylabel("Implied Volatility", fontsize=12)
        ax.set_title(title, fontsize=14, fontweight="bold")
        ax.grid(True, alpha=0.3)
        ax.legend()

        plt.tight_layout()
        return fig
