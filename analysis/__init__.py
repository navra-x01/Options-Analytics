"""
Analysis Module

This module provides PnL simulation and payoff analysis tools.
"""

from .pnl_simulation import simulate_pnl, plot_payoff_diagram, plot_greeks_approximation

__all__ = [
    "simulate_pnl",
    "plot_payoff_diagram",
    "plot_greeks_approximation",
]
