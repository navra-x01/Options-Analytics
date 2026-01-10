# Options Analytics Dashboard

A comprehensive Python-based **Options Analytics Tool** designed for demonstrating quantitative finance skills, particularly for **Quant / Quant Dev / Quant Research** roles.

This project implements a clean, mathematically correct options pricing and analysis system with interactive visualizations using Streamlit.

## 📋 Project Overview

This project provides a complete suite of options analytics tools including:

- **Black-Scholes Pricing**: Analytic pricing for European call and put options
- **Greeks Calculation**: Delta, Gamma, Theta, and Vega (both analytic and finite-difference)
- **Implied Volatility**: Robust IV computation from market prices using root-finding algorithms
- **Monte Carlo Simulation**: Option pricing using Geometric Brownian Motion with error estimation
- **Volatility Analysis**: Volatility smile and surface visualization from option chain data
- **Interactive Dashboard**: Streamlit-based web interface for exploration and analysis

## 🧮 Mathematical Background

### Black-Scholes Model

The Black-Scholes formula for a European call option is:

```
C = S·N(d₁) - K·e^(-rT)·N(d₂)
```

For a European put option:

```
P = K·e^(-rT)·N(-d₂) - S·N(-d₁)
```

where:

```
d₁ = (ln(S/K) + (r + σ²/2)·T) / (σ·√T)
d₂ = d₁ - σ·√T
```

- **S** = current spot price
- **K** = strike price
- **T** = time to maturity (years)
- **r** = risk-free interest rate (annualized)
- **σ** = volatility (annualized)
- **N(·)** = cumulative distribution function of standard normal distribution

### Greeks

The option Greeks measure sensitivity to various parameters:

- **Delta (Δ)**: Rate of change of option price with respect to spot price
  - Call: Δ = N(d₁)
  - Put: Δ = N(d₁) - 1

- **Gamma (Γ)**: Rate of change of delta with respect to spot price
  - Γ = n(d₁) / (S·σ·√T)

- **Theta (Θ)**: Rate of change of option price with respect to time
  - Call: Θ = -S·n(d₁)·σ/(2·√T) - r·K·e^(-rT)·N(d₂)
  - Put: Θ = -S·n(d₁)·σ/(2·√T) + r·K·e^(-rT)·N(-d₂)

- **Vega (ν)**: Rate of change of option price with respect to volatility
  - ν = S·n(d₁)·√T

### Implied Volatility

Implied volatility is computed by solving:

```
BS(S, K, T, r, σ_implied) = P_market
```

using numerical root-finding methods (Brent's method or bisection).

### Monte Carlo Simulation

Under the risk-neutral measure, stock prices follow Geometric Brownian Motion:

```
S_T = S_0 · exp((r - σ²/2)·T + σ·√T·Z)
```

where Z ~ N(0,1). The option price is then:

```
Price = e^(-rT) · E[max(S_T - K, 0)]  (call)
Price = e^(-rT) · E[max(K - S_T, 0)]  (put)
```

## 🚀 Installation

1. **Clone or download this repository**

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

   Required packages:
   - `numpy>=1.21.0`
   - `scipy>=1.7.0`
   - `pandas>=1.3.0`
   - `matplotlib>=3.4.0`
   - `plotly>=5.0.0`
   - `streamlit>=1.28.0`

## 📖 Usage

### Running the Dashboard

**Option 1: Using main.py**
```bash
python main.py --dashboard
```

**Option 2: Direct Streamlit command**
```bash
streamlit run dashboard/app.py
```

The dashboard will open in your default web browser at `http://localhost:8501`.

### Running Tests

```bash
python main.py --test
```

Or directly:
```bash
python -m pytest tests/
```

Or using unittest:
```bash
python -m unittest discover tests
```

### Using the Python API

```python
from pricing.black_scholes import black_scholes_call
from pricing.greeks import compute_all_greeks
from pricing.implied_vol import compute_implied_vol

# Price a call option
price = black_scholes_call(S=100, K=100, T=1.0, r=0.05, sigma=0.2)
print(f"Call price: ${price:.4f}")

# Compute Greeks
greeks = compute_all_greeks(S=100, K=100, T=1.0, r=0.05, sigma=0.2, option_type="call")
print(f"Delta: {greeks['delta']:.4f}")
print(f"Gamma: {greeks['gamma']:.4f}")

# Compute implied volatility
market_price = 10.45
iv = compute_implied_vol(market_price, S=100, K=100, T=1.0, r=0.05, option_type="call")
print(f"Implied volatility: {iv:.4f}")
```

## 📁 Project Structure

```
options_project/
├── data/
│   └── option_chain_sample.csv      # Sample option chain data (user-provided)
├── pricing/
│   ├── __init__.py
│   ├── black_scholes.py             # Black-Scholes pricing
│   ├── greeks.py                    # Greeks calculation
│   ├── implied_vol.py               # Implied volatility computation
│   └── monte_carlo.py               # Monte Carlo simulation
├── volatility/
│   ├── __init__.py
│   ├── smile.py                     # Volatility smile analysis
│   └── surface.py                    # Volatility surface analysis
├── dashboard/
│   ├── __init__.py
│   └── app.py                       # Streamlit dashboard
├── analysis/
│   ├── __init__.py
│   └── pnl_simulation.py            # PnL simulation and payoff diagrams
├── tests/
│   ├── __init__.py
│   ├── test_bs.py                   # Black-Scholes tests
│   ├── test_greeks.py               # Greeks tests
│   └── test_iv.py                   # Implied volatility tests
├── requirements.txt
├── README.md
└── main.py                          # Entry point
```

## ✨ Key Features

### 1. Single Option Analysis
- Black-Scholes pricing with intrinsic/time value breakdown
- Monte Carlo pricing with error estimation
- Complete Greeks calculation (Delta, Gamma, Theta, Vega)
- Comparison of analytic vs finite-difference Greeks
- Sensitivity analysis (price vs spot, volatility, time)
- Payoff diagrams

### 2. Volatility Analysis
- Implied volatility calculator
- Volatility smile visualization
- Volatility surface (3D and heatmap)

### 3. Option Chain Analysis
- Upload CSV with option chain data
- Batch IV computation
- Interactive volatility smile and surface plots

## 🧪 Testing

The project includes comprehensive unit tests:

- **test_bs.py**: Tests Black-Scholes pricing against known values, edge cases, and put-call parity
- **test_greeks.py**: Tests Greeks signs, magnitudes, and compares analytic vs finite-difference methods
- **test_iv.py**: Tests IV inversion accuracy and error handling

Run all tests:
```bash
python main.py --test
```

## 📊 Dashboard Features

The Streamlit dashboard provides:

1. **Interactive Input Controls**: Adjust spot price, strike, volatility, time to maturity, and risk-free rate in real-time
2. **Real-Time Calculations**: See option prices, Greeks, and Monte Carlo results update instantly
3. **Visualizations**: 
   - Sensitivity plots (price vs parameters)
   - Volatility smile and surface
   - Payoff diagrams
4. **Option Chain Upload**: Upload CSV files to analyze full option chains

## 🎯 What This Project Demonstrates

This project showcases skills relevant for **Quant / Quant Dev / Quant Research** roles:

✅ **Mathematical Rigor**: Correct implementation of Black-Scholes model and Greeks  
✅ **Numerical Methods**: Root-finding for IV, Monte Carlo simulation, finite-difference approximations  
✅ **Software Engineering**: Clean code structure, type hints, comprehensive testing  
✅ **Data Analysis**: Option chain processing, volatility analysis  
✅ **Visualization**: Interactive dashboards, 3D plots, sensitivity analysis  
✅ **Problem-Solving**: Edge case handling, error management, numerical stability  

## 📝 Notes

- **No Live Trading**: This is an analytics tool only - no execution or trading functionality
- **European Options Only**: The implementation focuses on European-style options
- **Sample Data**: You'll need to provide your own `option_chain_sample.csv` file in the `data/` directory
- **CSV Format**: Option chain CSV should have columns: `strike`, `maturity`, `option_type`, `market_price`

## 🔧 Future Enhancements

Potential additions for extended learning:

- American option pricing (binomial tree, finite difference)
- Exotic options (barriers, Asian, etc.)
- Volatility models (local volatility, stochastic volatility)
- Portfolio Greeks and risk management
- Historical volatility calculation
- Option strategies (spreads, straddles, etc.)

## 📄 License

This project is provided as-is for educational and demonstration purposes.

## 👤 Author

Built for quant role applications and interviews.

---

**Happy Analyzing! 📈**
