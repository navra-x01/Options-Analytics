# Option Chain Data

Place your option chain CSV files in this directory.

## Expected CSV Format

The CSV file should contain the following columns:

- **strike**: Strike price (float)
- **maturity**: Maturity date (datetime string like "2024-12-31") or time to maturity in years (float)
- **option_type**: "call" or "put" (string)
- **market_price**: Observed market price of the option (float)

## Example CSV Format

```csv
strike,maturity,option_type,market_price
90,2024-12-31,call,15.50
95,2024-12-31,call,12.30
100,2024-12-31,call,10.45
105,2024-12-31,call,8.20
110,2024-12-31,call,6.10
90,2024-12-31,put,2.10
95,2024-12-31,put,3.50
100,2024-12-31,put,5.57
105,2024-12-31,put,8.20
110,2024-12-31,put,11.30
```

## Alternative Format (Time to Maturity in Years)

```csv
strike,maturity,option_type,market_price
90,1.0,call,15.50
95,1.0,call,12.30
100,1.0,call,10.45
105,1.0,call,8.20
110,1.0,call,6.10
```

## Notes

- Maturity can be specified as a datetime string (will be converted to years from current date)
- Maturity can also be specified directly in years (float)
- The dashboard will automatically compute implied volatilities for all options in the chain
- Missing or invalid prices will result in NaN in the implied_vol column
