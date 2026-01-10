# Quick Start Guide

## Installation

1. Install dependencies:
   ```bash
   cd options_project
   pip install -r requirements.txt
   ```

## Running the Dashboard

```bash
python main.py --dashboard
```

Or directly:
```bash
streamlit run dashboard/app.py
```

## Running Tests

```bash
python main.py --test
```

Or using unittest:
```bash
cd options_project
python -m unittest discover tests
```

## Quick Python Example

```python
# Make sure you're in the options_project directory or add it to PYTHONPATH
from pricing.black_scholes import black_scholes_call
from pricing.greeks import compute_all_greeks

# Price a call option
price = black_scholes_call(S=100, K=100, T=1.0, r=0.05, sigma=0.2)
print(f"Call price: ${price:.4f}")

# Get Greeks
greeks = compute_all_greeks(S=100, K=100, T=1.0, r=0.05, sigma=0.2, option_type="call")
print(f"Delta: {greeks['delta']:.4f}")
print(f"Gamma: {greeks['gamma']:.4f}")
print(f"Theta: {greeks['theta']:.4f}")
print(f"Vega: {greeks['vega']:.4f}")
```

## Troubleshooting

### Import Errors

If you get import errors when running tests or scripts:

1. Make sure you're in the `options_project` directory
2. Or add the project root to PYTHONPATH:
   ```bash
   export PYTHONPATH="${PYTHONPATH}:$(pwd)"
   ```
   (Windows: `set PYTHONPATH=%PYTHONPATH%;%CD%`)

### Dashboard Not Starting

- Make sure Streamlit is installed: `pip install streamlit`
- Check that you're running from the correct directory
- Try: `streamlit run dashboard/app.py --server.port 8501`

### Test Failures

- Ensure all dependencies are installed
- Run tests from the `options_project` directory
- Check Python version (3.7+ recommended)
