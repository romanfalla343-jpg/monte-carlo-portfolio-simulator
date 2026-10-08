# Monte Carlo Portfolio Risk Simulator

Simulates portfolio return distributions using Monte Carlo methods on real FTSE 100 data.

## What it does

- Pulls 3 years of live price data via Yahoo Finance API
- Runs 10,000 correlated simulations using Cholesky decomposition to preserve asset correlations
- Computes Value at Risk (VaR), Conditional VaR (CVaR), and Sharpe Ratio
- Plots an approximate efficient frontier from 5,000 randomly weighted portfolios (no optimiser)
- Outputs a 5-chart dashboard saved as PNG

## Sample output (May 2026)

| Metric | Value |
|--------|-------|
| Annualised Return | 10.94% |
| Annualised Volatility | 12.75% |
| Sharpe Ratio | 0.51 |
| VaR (95%, 1yr) | £10,218 |
| CVaR (95%, 1yr) | £14,617 |
| Median Terminal Value | £110,424 |

## Usage

```bash
pip install yfinance pandas numpy matplotlib
python monte_carlo_portfolio_simulator.py
```

## Methodology

Correlated daily asset returns are simulated as normal draws (arithmetic returns compounded over 252 days; not log-normal GBM). The Cholesky decomposition of the historical correlation matrix is applied to standard normal random draws to preserve real-world asset correlations. VaR is computed as the 5th percentile of the terminal return distribution. CVaR is the mean of returns below the VaR threshold.
