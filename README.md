# Monte Carlo Portfolio Risk Simulator

Simulates portfolio return distributions using Monte Carlo methods on real FTSE 100 data.

## What it does

- Pulls 3 years of current historical price data via Yahoo Finance API
- Runs 10,000 correlated simulations using Cholesky decomposition to preserve asset correlations
- Computes Value at Risk (VaR), Conditional VaR (CVaR), and Sharpe Ratio
- Plots an approximate efficient frontier from 5,000 randomly weighted portfolios (no optimiser)
- Outputs a 5-chart dashboard saved as PNG

## Sample output (May 2026)

**Initial portfolio: £100,000**

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

## Example Output

![Monte Carlo Portfolio Dashboard](monte_carlo_results_2026-10-08.png)
