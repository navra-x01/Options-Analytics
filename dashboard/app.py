"""
Streamlit Dashboard for Options Analytics

Interactive dashboard for exploring option pricing, Greeks, implied volatility,
and volatility surfaces.
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sys
import os


def main() -> None:
    """
    Main entry point for the Options Analytics dashboard.

    NOTE: This function is called on every Streamlit rerun.
    Do not execute Streamlit commands at module import time, otherwise
    reruns (triggered by widget interaction) may result in a blank page.
    """

    # Add parent directory to path for imports
    try:
        parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        if parent_dir not in sys.path:
            sys.path.insert(0, parent_dir)
    except Exception:
        # Fallback: try adding current working directory
        try:
            cwd = os.getcwd()
            if cwd not in sys.path:
                sys.path.insert(0, cwd)
        except Exception:
            pass

    # Import required modules with error handling
    try:
        from pricing.black_scholes import black_scholes_call, black_scholes_put
        from pricing.greeks import compute_all_greeks, compute_greeks_fd
        from pricing.implied_vol import compute_implied_vol, compute_iv_chain
        from pricing.monte_carlo import monte_carlo_european, compare_mc_bs
        from volatility.smile import build_volatility_smile, plot_volatility_smile
        from volatility.surface import build_volatility_surface, plot_volatility_surface
        from analysis.pnl_simulation import plot_payoff_diagram, plot_greeks_approximation
    except ImportError as e:
        st.error(f"Import Error: {e}")
        st.error("Please ensure all dependencies are installed: pip install -r requirements.txt")
        st.stop()

    # Page configuration
    st.set_page_config(
        page_title="Options Analytics Dashboard",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # Title
    st.title("📊 Options Analytics Dashboard")
    st.markdown(
        """
        A comprehensive tool for analyzing European options using Black-Scholes pricing,
        Greeks calculation, implied volatility, and Monte Carlo simulation.
        """
    )

    # Sidebar inputs
    st.sidebar.header("Input Parameters")

    S = st.sidebar.number_input(
        "Spot Price (S)",
        min_value=1.0,
        max_value=1000.0,
        value=100.0,
        step=1.0,
        help="Current price of the underlying asset",
    )

    K = st.sidebar.number_input(
        "Strike Price (K)",
        min_value=1.0,
        max_value=1000.0,
        value=100.0,
        step=1.0,
        help="Strike price of the option",
    )

    T = st.sidebar.number_input(
        "Time to Maturity (T)",
        min_value=0.01,
        max_value=10.0,
        value=1.0,
        step=0.1,
        format="%.2f",
        help="Time to expiration in years",
    )

    r = st.sidebar.number_input(
        "Risk-Free Rate (r)",
        min_value=0.0,
        max_value=1.0,
        value=0.05,
        step=0.01,
        format="%.4f",
        help="Annualized risk-free interest rate",
    )

    sigma = st.sidebar.number_input(
        "Volatility (σ)",
        min_value=0.01,
        max_value=2.0,
        value=0.2,
        step=0.01,
        format="%.4f",
        help="Annualized volatility",
    )

    option_type = st.sidebar.selectbox(
        "Option Type",
        ["call", "put"],
        index=0,
        help="European call or put option",
    )

    # Calculate Black-Scholes price (needed across multiple tabs)
    try:
        if option_type == "call":
            bs_price = black_scholes_call(S, K, T, r, sigma)
        else:
            bs_price = black_scholes_put(S, K, T, r, sigma)
    except Exception as e:
        st.error(f"Error calculating option price: {e}")
        bs_price = 0.0

    # Main tabs
    tab1, tab2, tab3 = st.tabs(
        ["Single Option Analysis", "Volatility Analysis", "Option Chain Analysis"]
    )

    # Tab 1: Single Option Analysis
    with tab1:
        st.header("Single Option Analysis")

        # Pricing section
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Black-Scholes Pricing")

            st.metric("Option Price", f"${bs_price:.4f}")

            # Intrinsic and time value
            if option_type == "call":
                intrinsic = max(S - K, 0.0)
            else:
                intrinsic = max(K - S, 0.0)

            time_value = bs_price - intrinsic * np.exp(-r * T)
            st.metric("Intrinsic Value", f"${intrinsic:.4f}")
            st.metric("Time Value", f"${time_value:.4f}")

        with col2:
            st.subheader("Monte Carlo Pricing")
            n_simulations = st.number_input(
                "Number of Simulations",
                min_value=1000,
                max_value=1000000,
                value=50000,  # Reduced default for faster computation
                step=10000,
                key="mc_sims",
            )

            try:
                mc_price, std_error = monte_carlo_european(
                    S, K, T, r, sigma, option_type, n_simulations=int(n_simulations)
                )

                st.metric("MC Price", f"${mc_price:.4f}")
                st.metric("Standard Error", f"${std_error:.6f}")
                st.metric("Error vs BS", f"${abs(mc_price - bs_price):.6f}")
            except Exception as e:
                st.error(f"Error in Monte Carlo simulation: {e}")
                st.info("Try reducing the number of simulations or adjusting parameters.")

        # Greeks section
        st.subheader("Greeks")
        try:
            greeks = compute_all_greeks(S, K, T, r, sigma, option_type)

            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Delta (Δ)", f"{greeks['delta']:.4f}")
            with col2:
                st.metric("Gamma (Γ)", f"{greeks['gamma']:.4f}")
            with col3:
                st.metric("Theta (Θ)", f"{greeks['theta']:.4f}")
            with col4:
                st.metric("Vega (ν)", f"{greeks['vega']:.4f}")
        except Exception as e:
            st.error(f"Error calculating Greeks: {e}")
            greeks = {"delta": 0.0, "gamma": 0.0, "theta": 0.0, "vega": 0.0}

        # Greeks comparison table
        st.subheader("Analytic vs Finite-Difference Greeks")
        try:
            greeks_fd = compute_greeks_fd(S, K, T, r, sigma, option_type)
        except Exception as e:
            st.error(f"Error calculating finite-difference Greeks: {e}")
            greeks_fd = {"delta": 0.0, "gamma": 0.0, "theta": 0.0, "vega": 0.0}

        try:
            comparison_df = pd.DataFrame(
                {
                    "Greek": ["Delta", "Gamma", "Theta", "Vega"],
                    "Analytic": [
                        greeks.get("delta", 0.0),
                        greeks.get("gamma", 0.0),
                        greeks.get("theta", 0.0),
                        greeks.get("vega", 0.0),
                    ],
                    "Finite-Diff": [
                        greeks_fd.get("delta", 0.0),
                        greeks_fd.get("gamma", 0.0),
                        greeks_fd.get("theta", 0.0),
                        greeks_fd.get("vega", 0.0),
                    ],
                    "Difference": [
                        abs(greeks.get("delta", 0.0) - greeks_fd.get("delta", 0.0)),
                        abs(greeks.get("gamma", 0.0) - greeks_fd.get("gamma", 0.0)),
                        abs(greeks.get("theta", 0.0) - greeks_fd.get("theta", 0.0)),
                        abs(greeks.get("vega", 0.0) - greeks_fd.get("vega", 0.0)),
                    ],
                }
            )
            st.dataframe(comparison_df, use_container_width=True)
        except Exception as e:
            st.error(f"Error displaying Greeks comparison: {e}")

        # Sensitivity plots
        st.subheader("Sensitivity Analysis")

        plot_type = st.selectbox(
            "Sensitivity to:",
            ["Spot Price (S)", "Volatility (σ)", "Time to Maturity (T)"],
        )

        try:
            if plot_type == "Spot Price (S)":
                spot_range = np.linspace(max(0.5 * S, 1.0), 1.5 * S, 100)
                prices = []
                for s in spot_range:
                    try:
                        if option_type == "call":
                            prices.append(black_scholes_call(s, K, T, r, sigma))
                        else:
                            prices.append(black_scholes_put(s, K, T, r, sigma))
                    except Exception:
                        prices.append(0.0)

                fig = go.Figure()
                fig.add_trace(
                    go.Scatter(
                        x=spot_range,
                        y=prices,
                        mode="lines",
                        name="Option Price",
                        line=dict(width=2),
                    )
                )
                fig.add_vline(
                    x=S, line_dash="dash", line_color="red", annotation_text="Current Spot"
                )
                fig.update_layout(
                    title="Option Price vs Spot Price",
                    xaxis_title="Spot Price",
                    yaxis_title="Option Price",
                    template="plotly_white",
                )
                st.plotly_chart(fig, use_container_width=True)

            elif plot_type == "Volatility (σ)":
                vol_range = np.linspace(0.01, 1.0, 100)
                prices = []
                for vol in vol_range:
                    try:
                        if option_type == "call":
                            prices.append(black_scholes_call(S, K, T, r, vol))
                        else:
                            prices.append(black_scholes_put(S, K, T, r, vol))
                    except Exception:
                        prices.append(0.0)

                fig = go.Figure()
                fig.add_trace(
                    go.Scatter(
                        x=vol_range,
                        y=prices,
                        mode="lines",
                        name="Option Price",
                        line=dict(width=2),
                    )
                )
                fig.add_vline(
                    x=sigma,
                    line_dash="dash",
                    line_color="red",
                    annotation_text="Current Volatility",
                )
                fig.update_layout(
                    title="Option Price vs Volatility",
                    xaxis_title="Volatility",
                    yaxis_title="Option Price",
                    template="plotly_white",
                )
                st.plotly_chart(fig, use_container_width=True)

            else:  # Time to Maturity
                T_range = np.linspace(0.01, max(T * 2, 0.1), 100)
                prices = []
                for t in T_range:
                    try:
                        if option_type == "call":
                            prices.append(black_scholes_call(S, K, t, r, sigma))
                        else:
                            prices.append(black_scholes_put(S, K, t, r, sigma))
                    except Exception:
                        prices.append(0.0)

                fig = go.Figure()
                fig.add_trace(
                    go.Scatter(
                        x=T_range,
                        y=prices,
                        mode="lines",
                        name="Option Price",
                        line=dict(width=2),
                    )
                )
                fig.add_vline(
                    x=T,
                    line_dash="dash",
                    line_color="red",
                    annotation_text="Current Maturity",
                )
                fig.update_layout(
                    title="Option Price vs Time to Maturity",
                    xaxis_title="Time to Maturity (years)",
                    yaxis_title="Option Price",
                    template="plotly_white",
                )
                st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.error(f"Error generating sensitivity plot: {e}")

        # Payoff diagram
        st.subheader("Payoff Diagram")
        if st.button("Generate Payoff Diagram"):
            try:
                import matplotlib

                matplotlib.use("Agg")  # Use non-interactive backend
                import matplotlib.pyplot as plt

                fig = plot_payoff_diagram(S, K, T, r, sigma, option_type)
                st.pyplot(fig)
                plt.close(fig)  # Close figure to free memory
            except Exception as e:
                st.error(f"Error generating payoff diagram: {e}")

    # Tab 2: Volatility Analysis
    with tab2:
        st.header("Volatility Analysis")

        st.subheader("Implied Volatility Calculator")
        col1, col2 = st.columns(2)

        with col1:
            # Use bs_price if available, otherwise default to 10.0
            default_price = bs_price if bs_price > 0 else 10.0
            market_price = st.number_input(
                "Market Price",
                min_value=0.01,
                max_value=1000.0,
                value=default_price,
                step=0.01,
                help="Observed market price of the option",
            )

        with col2:
            if st.button("Compute Implied Volatility"):
                try:
                    iv = compute_implied_vol(market_price, S, K, T, r, option_type)
                    st.metric("Implied Volatility", f"{iv:.4f} ({iv*100:.2f}%)")

                    # Compare with input volatility
                    diff = abs(iv - sigma)
                    st.metric("Difference from Input Vol", f"{diff:.4f}")

                    # Re-price to validate
                    if option_type == "call":
                        repriced = black_scholes_call(S, K, T, r, iv)
                    else:
                        repriced = black_scholes_put(S, K, T, r, iv)

                    st.metric("Re-priced Option Value", f"${repriced:.4f}")
                    st.metric("Pricing Error", f"${abs(repriced - market_price):.6f}")

                except ValueError as e:
                    st.error(f"Error computing IV: {e}")

        st.subheader("Volatility Smile (Sample Data)")
        st.info(
            "Upload option chain data in Tab 3 to see volatility smile and surface visualizations."
        )

    # Tab 3: Option Chain Analysis
    with tab3:
        st.header("Option Chain Analysis")

        uploaded_file = st.file_uploader(
            "Upload Option Chain CSV",
            type=["csv"],
            help="CSV file with columns: strike, maturity, option_type, market_price",
        )

        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)

                st.subheader("Option Chain Data")
                st.dataframe(df.head(20), use_container_width=True)

                # Check required columns
                required_cols = ["strike", "maturity", "option_type", "market_price"]
                missing_cols = [col for col in required_cols if col not in df.columns]

                if missing_cols:
                    st.error(f"Missing required columns: {missing_cols}")
                else:
                    # Compute IVs
                    st.subheader("Computing Implied Volatilities...")
                    df_with_iv = compute_iv_chain(df, S, r)

                    st.subheader("Option Chain with Implied Volatilities")
                    st.dataframe(df_with_iv, use_container_width=True)

                    # Volatility Smile
                    st.subheader("Volatility Smile")
                    if "maturity" in df.columns:
                        # Get unique maturities
                        if pd.api.types.is_datetime64_any_dtype(df["maturity"]):
                            maturities = sorted(df["maturity"].unique())
                        else:
                            maturities = sorted(df["maturity"].unique())

                        selected_maturity = st.selectbox(
                            "Select Maturity", maturities, key="smile_maturity"
                        )

                        try:
                            smile_df = build_volatility_smile(
                                df_with_iv, selected_maturity, S, r
                            )

                            # Separate calls and puts
                            calls = smile_df[
                                smile_df["option_type"].str.lower() == "call"
                            ]
                            puts = smile_df[smile_df["option_type"].str.lower() == "put"]

                            fig = go.Figure()

                            if len(calls) > 0:
                                fig.add_trace(
                                    go.Scatter(
                                        x=calls["strike"],
                                        y=calls["implied_vol"],
                                        mode="lines+markers",
                                        name="Call IV",
                                        line=dict(width=2),
                                    )
                                )

                            if len(puts) > 0:
                                fig.add_trace(
                                    go.Scatter(
                                        x=puts["strike"],
                                        y=puts["implied_vol"],
                                        mode="lines+markers",
                                        name="Put IV",
                                        line=dict(width=2),
                                    )
                                )

                            fig.add_vline(
                                x=S,
                                line_dash="dash",
                                line_color="red",
                                annotation_text="Spot Price",
                            )

                            fig.update_layout(
                                title=f"Volatility Smile (Maturity: {selected_maturity})",
                                xaxis_title="Strike Price",
                                yaxis_title="Implied Volatility",
                                template="plotly_white",
                            )

                            st.plotly_chart(fig, use_container_width=True)

                        except Exception as e:
                            st.error(f"Error building volatility smile: {e}")

                    # Volatility Surface
                    st.subheader("Volatility Surface")
                    option_type_filter = st.selectbox(
                        "Filter by Option Type",
                        ["All", "call", "put"],
                        key="surface_filter",
                    )

                    if st.button("Build Volatility Surface"):
                        try:
                            filter_type = (
                                None if option_type_filter == "All" else option_type_filter
                            )
                            strikes, maturities, iv_matrix = build_volatility_surface(
                                df_with_iv, S, r, option_type_filter=filter_type
                            )

                            # 3D Surface Plot
                            fig = plot_volatility_surface(strikes, maturities, iv_matrix)
                            st.plotly_chart(fig, use_container_width=True)

                        except Exception as e:
                            st.error(f"Error building volatility surface: {e}")

            except Exception as e:
                st.error(f"Error reading CSV file: {e}")

        else:
            st.info("Please upload a CSV file with option chain data to begin analysis.")
            st.markdown(
                """
            **Expected CSV Format:**
            - `strike`: Strike price
            - `maturity`: Maturity date (datetime string) or time to maturity in years
            - `option_type`: "call" or "put"
            - `market_price`: Observed market price
            """
            )

    # Footer
    st.sidebar.markdown("---")
    st.sidebar.markdown(
        """
        **Options Analytics Dashboard**

        Features:
        - Black-Scholes pricing
        - Greeks calculation
        - Implied volatility
        - Monte Carlo simulation
        - Volatility analysis
        """
    )


if __name__ == "__main__":
    # Allow running `python dashboard/app.py` locally for debugging.
    main()
