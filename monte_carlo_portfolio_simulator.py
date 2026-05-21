"""
Monte Carlo Portfolio Risk Simulator
=====================================
Simulates portfolio return distributions using Monte Carlo methods.
Computes Value at Risk (VaR), Conditional VaR (CVaR), Sharpe Ratio,
and optimal portfolio weights via mean-variance optimisation.

Author: Roman Falla
GitHub: github.com/romanfalla343-jpg

Dependencies:
    pip install yfinance pandas numpy matplotlib scipy

Usage:
    python monte_carlo_portfolio_simulator.py
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import yfinance as yf
import datetime
import warnings

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

# Default portfolio — FTSE 100 blue chips
DEFAULT_TICKERS = {
    "SHEL.L":  "Shell",
    "AZN.L":   "AstraZeneca",
    "HSBA.L":  "HSBC",
    "ULVR.L":  "Unilever",
    "LGEN.L":  "Legal & General",
}

DEFAULT_WEIGHTS = [0.25, 0.25, 0.20, 0.20, 0.10]   # must sum to 1.0

SIMULATION_CONFIG = {
    "n_simulations":    10_000,   # number of Monte Carlo paths
    "horizon_days":     252,      # 1-year holding period
    "initial_value":    100_000,  # £100,000 portfolio
    "lookback_years":   3,        # years of historical data for calibration
    "confidence_var":   0.95,     # VaR confidence level
    "risk_free_rate":   0.045,    # UK risk-free rate (gilt yield)
    "n_random_ports":   5_000,    # portfolios for efficient frontier
}

# ─────────────────────────────────────────────────────────────────────────────
# DATA
# ─────────────────────────────────────────────────────────────────────────────

def fetch_price_data(tickers: dict, lookback_years: int) -> pd.DataFrame:
    today = datetime.date.today()
    start = today - datetime.timedelta(days=lookback_years * 365 + 30)
    print(f"\nFetching {lookback_years} years of price data for {len(tickers)} assets...")

    prices = {}
    for ticker, name in tickers.items():
        try:
            data = yf.download(ticker, start=start.strftime("%Y-%m-%d"),
                               end=today.strftime("%Y-%m-%d"),
                               progress=False, auto_adjust=True)
            if not data.empty:
                prices[name] = data["Close"].squeeze()
                print(f"  ✓ {name} ({ticker}): {len(data)} trading days")
            else:
                print(f"  ✗ {name} ({ticker}): no data")
        except Exception as e:
            print(f"  ✗ {name} ({ticker}): {e}")

    df = pd.DataFrame(prices).dropna()
    print(f"\n  Dataset: {len(df)} common trading days\n")
    return df


def compute_returns(prices: pd.DataFrame) -> pd.DataFrame:
    return prices.pct_change().dropna()


# ─────────────────────────────────────────────────────────────────────────────
# MONTE CARLO ENGINE
# ─────────────────────────────────────────────────────────────────────────────

def run_monte_carlo(returns: pd.DataFrame, weights: np.ndarray,
                    config: dict) -> dict:
    """
    Simulate portfolio paths using correlated Geometric Brownian Motion.
    Uses Cholesky decomposition to preserve asset correlations.
    """
    n_assets  = len(weights)
    n_sims    = config["n_simulations"]
    horizon   = config["horizon_days"]
    port_val  = config["initial_value"]
    conf      = config["confidence_var"]

    mu    = returns.mean().values           # daily mean returns
    sigma = returns.std().values            # daily volatility
    corr  = returns.corr().values           # correlation matrix
    cov   = returns.cov().values * 252      # annualised covariance

    # Portfolio statistics (annualised)
    port_return = np.dot(weights, mu) * 252
    port_vol    = np.sqrt(weights @ cov @ weights)
    sharpe      = (port_return - config["risk_free_rate"]) / port_vol

    # Cholesky decomposition for correlated random draws
    L = np.linalg.cholesky(corr)

    print(f"Running {n_sims:,} Monte Carlo simulations over {horizon} days...")

    # Simulate n_sims paths of horizon days
    # Shape: (n_sims, horizon, n_assets)
    Z = np.random.standard_normal((n_sims, horizon, n_assets))
    Z_corr = Z @ L.T                         # apply correlation structure

    # Daily returns for each simulation
    daily_ret = mu + sigma * Z_corr          # (n_sims, horizon, n_assets)

    # Portfolio daily returns
    port_daily = daily_ret @ weights         # (n_sims, horizon)

    # Cumulative portfolio value paths
    cum_ret  = np.cumprod(1 + port_daily, axis=1)     # (n_sims, horizon)
    paths    = port_val * cum_ret                      # £ value paths

    # Terminal values
    final_values  = paths[:, -1]
    final_returns = final_values / port_val - 1

    # Value at Risk & CVaR
    var_threshold = np.percentile(final_returns, (1 - conf) * 100)
    cvar          = final_returns[final_returns <= var_threshold].mean()
    var_gbp       = port_val * abs(var_threshold)
    cvar_gbp      = port_val * abs(cvar)

    # Percentile paths
    p5  = np.percentile(paths, 5,  axis=0)
    p25 = np.percentile(paths, 25, axis=0)
    p50 = np.percentile(paths, 50, axis=0)
    p75 = np.percentile(paths, 75, axis=0)
    p95 = np.percentile(paths, 95, axis=0)

    print(f"\n{'='*55}")
    print(f"  PORTFOLIO STATISTICS")
    print(f"{'='*55}")
    print(f"  Assets            : {', '.join(returns.columns.tolist())}")
    print(f"  Weights           : {[f'{w:.0%}' for w in weights]}")
    print(f"  Ann. Return       : {port_return:.2%}")
    print(f"  Ann. Volatility   : {port_vol:.2%}")
    print(f"  Sharpe Ratio      : {sharpe:.2f}")
    print(f"  VaR ({conf:.0%}, 1yr)  : £{var_gbp:,.0f}  ({abs(var_threshold):.2%})")
    print(f"  CVaR ({conf:.0%}, 1yr) : £{cvar_gbp:,.0f}  ({abs(cvar):.2%})")
    print(f"  Median Terminal   : £{np.median(final_values):,.0f}")
    print(f"  5th Pct Terminal  : £{np.percentile(final_values,5):,.0f}")
    print(f"  95th Pct Terminal : £{np.percentile(final_values,95):,.0f}")
    print(f"{'='*55}\n")

    return {
        "paths":          paths,
        "final_values":   final_values,
        "final_returns":  final_returns,
        "p5": p5, "p25": p25, "p50": p50, "p75": p75, "p95": p95,
        "port_return":    port_return,
        "port_vol":       port_vol,
        "sharpe":         sharpe,
        "var_threshold":  var_threshold,
        "cvar":           cvar,
        "var_gbp":        var_gbp,
        "cvar_gbp":       cvar_gbp,
        "cov":            cov,
        "mu_annual":      mu * 252,
    }


# ─────────────────────────────────────────────────────────────────────────────
# EFFICIENT FRONTIER
# ─────────────────────────────────────────────────────────────────────────────

def compute_efficient_frontier(mu_annual: np.ndarray, cov: np.ndarray,
                                rf: float, n_ports: int) -> pd.DataFrame:
    """Generate random portfolio weights to approximate the efficient frontier."""
    n_assets = len(mu_annual)
    results  = np.zeros((n_ports, 3 + n_assets))

    for i in range(n_ports):
        w = np.random.dirichlet(np.ones(n_assets))
        r = np.dot(w, mu_annual)
        v = np.sqrt(w @ cov @ w)
        s = (r - rf) / v
        results[i, :3]  = [r, v, s]
        results[i, 3:]  = w

    cols = ["Return", "Volatility", "Sharpe"] + [f"w_{i}" for i in range(n_assets)]
    return pd.DataFrame(results, columns=cols)


# ─────────────────────────────────────────────────────────────────────────────
# VISUALISATION
# ─────────────────────────────────────────────────────────────────────────────

def plot_results(results: dict, returns: pd.DataFrame, weights: np.ndarray,
                 tickers: dict, config: dict):

    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.size": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.facecolor": "white",
        "axes.facecolor": "#FAFAFA",
    })

    fig = plt.figure(figsize=(18, 12))
    fig.suptitle("Monte Carlo Portfolio Risk Simulator  —  Roman Falla",
                 fontsize=16, fontweight="bold", y=0.98)

    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.4, wspace=0.35)
    ax1 = fig.add_subplot(gs[0, :2])   # simulation paths (wide)
    ax2 = fig.add_subplot(gs[0, 2])    # terminal value distribution
    ax3 = fig.add_subplot(gs[1, 0])    # efficient frontier
    ax4 = fig.add_subplot(gs[1, 1])    # correlation heatmap
    ax5 = fig.add_subplot(gs[1, 2])    # individual asset returns

    horizon = config["horizon_days"]
    port_val = config["initial_value"]
    days = np.arange(horizon)

    # ── Chart 1: Simulation paths with percentile bands ────────────────────
    n_show = min(200, len(results["paths"]))
    idx    = np.random.choice(len(results["paths"]), n_show, replace=False)
    for i in idx:
        ax1.plot(days, results["paths"][i], color="#2E86AB", alpha=0.03, linewidth=0.5)

    ax1.fill_between(days, results["p5"],  results["p95"], alpha=0.15,
                     color="#F18F01", label="5th–95th percentile")
    ax1.fill_between(days, results["p25"], results["p75"], alpha=0.25,
                     color="#F18F01", label="25th–75th percentile")
    ax1.plot(days, results["p50"], color="#A23B72", linewidth=2.5,
             label=f"Median: £{results['p50'][-1]:,.0f}")
    ax1.plot(days, results["p5"],  color="#E74C3C", linewidth=1.5, linestyle="--",
             label=f"5th pct: £{results['p5'][-1]:,.0f}")
    ax1.axhline(port_val, color="grey", linestyle=":", linewidth=1.2,
                label=f"Initial: £{port_val:,}")

    ax1.set_title(f"Monte Carlo Simulation — {config['n_simulations']:,} Paths  "
                  f"(1-Year Horizon)", fontweight="bold")
    ax1.set_xlabel("Trading Days")
    ax1.set_ylabel("Portfolio Value (£)")
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"£{x/1000:.0f}k"))
    ax1.legend(fontsize=9, loc="upper left")

    # ── Chart 2: Terminal value distribution with VaR ──────────────────────
    ax2.hist(results["final_values"], bins=80, color="#2E86AB",
             edgecolor="white", linewidth=0.3, alpha=0.85)
    var_val = port_val * (1 + results["var_threshold"])
    ax2.axvline(var_val, color="#E74C3C", linewidth=2,
                label=f"VaR ({config['confidence_var']:.0%}): £{results['var_gbp']:,.0f}")
    ax2.axvline(np.median(results["final_values"]), color="#A23B72",
                linewidth=2, linestyle="--",
                label=f"Median: £{np.median(results['final_values']):,.0f}")
    ax2.set_title("Terminal Portfolio Value Distribution", fontweight="bold")
    ax2.set_xlabel("Portfolio Value (£)")
    ax2.set_ylabel("Frequency")
    ax2.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"£{x/1000:.0f}k"))
    ax2.legend(fontsize=8)

    # ── Chart 3: Efficient frontier ────────────────────────────────────────
    ef = compute_efficient_frontier(
        results["mu_annual"], results["cov"],
        config["risk_free_rate"], config["n_random_ports"]
    )
    sc = ax3.scatter(ef["Volatility"], ef["Return"],
                     c=ef["Sharpe"], cmap="RdYlGn", s=4, alpha=0.6)
    plt.colorbar(sc, ax=ax3, label="Sharpe Ratio")
    ax3.scatter(results["port_vol"], results["port_return"],
                color="#E74C3C", s=120, zorder=5, marker="*",
                label=f"Current portfolio\nSharpe: {results['sharpe']:.2f}")
    max_sharpe = ef.loc[ef["Sharpe"].idxmax()]
    ax3.scatter(max_sharpe["Volatility"], max_sharpe["Return"],
                color="#A23B72", s=120, zorder=5, marker="D",
                label=f"Max Sharpe\nSharpe: {max_sharpe['Sharpe']:.2f}")
    ax3.set_title("Efficient Frontier", fontweight="bold")
    ax3.set_xlabel("Annual Volatility")
    ax3.set_ylabel("Annual Return")
    ax3.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0%}"))
    ax3.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0%}"))
    ax3.legend(fontsize=8)

    # ── Chart 4: Correlation heatmap ───────────────────────────────────────
    corr = returns.corr()
    names = list(tickers.values())
    im = ax4.imshow(corr.values, cmap="RdYlGn", vmin=-1, vmax=1, aspect="auto")
    plt.colorbar(im, ax=ax4)
    ax4.set_xticks(range(len(names)))
    ax4.set_yticks(range(len(names)))
    ax4.set_xticklabels(names, rotation=30, ha="right", fontsize=8)
    ax4.set_yticklabels(names, fontsize=8)
    for i in range(len(names)):
        for j in range(len(names)):
            ax4.text(j, i, f"{corr.values[i,j]:.2f}",
                     ha="center", va="center", fontsize=8,
                     color="black" if abs(corr.values[i,j]) < 0.7 else "white")
    ax4.set_title("Asset Correlation Matrix", fontweight="bold")

    # ── Chart 5: Individual asset annualised returns ───────────────────────
    ann_ret  = returns.mean() * 252
    ann_vol  = returns.std() * np.sqrt(252)
    colours  = ["#2E86AB" if r > 0 else "#E74C3C" for r in ann_ret]
    bars = ax5.bar(names, ann_ret.values * 100, color=colours, width=0.5,
                   edgecolor="white")
    ax5.errorbar(names, ann_ret.values * 100,
                 yerr=ann_vol.values * 100, fmt="none",
                 color="black", capsize=4, linewidth=1.5)
    ax5.axhline(0, color="grey", linewidth=0.8)
    ax5.set_title("Annualised Returns ± 1σ Volatility", fontweight="bold")
    ax5.set_ylabel("Return (%)")
    ax5.tick_params(axis="x", rotation=20)
    for bar, val in zip(bars, ann_ret.values * 100):
        ax5.text(bar.get_x() + bar.get_width()/2,
                 bar.get_height() + (1 if val >= 0 else -3),
                 f"{val:.1f}%", ha="center", fontsize=8)

    out = f"monte_carlo_results_{datetime.date.today()}.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Chart saved: {out}")
    plt.show()


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def run_simulator(tickers=None, weights=None, config=None):
    if tickers is None:  tickers = DEFAULT_TICKERS
    if weights is None:  weights = np.array(DEFAULT_WEIGHTS)
    if config  is None:  config  = SIMULATION_CONFIG

    assert abs(sum(weights) - 1.0) < 1e-6, "Weights must sum to 1.0"
    assert len(weights) == len(tickers),    "Weights and tickers must match"

    np.random.seed(42)

    print("=" * 55)
    print("  MONTE CARLO PORTFOLIO RISK SIMULATOR")
    print(f"  Run date : {datetime.date.today().strftime('%d %B %Y')}")
    print("=" * 55)

    prices  = fetch_price_data(tickers, config["lookback_years"])

    # Align weights to available assets
    available = [name for name in tickers.values() if name in prices.columns]
    available_tickers = {k: v for k, v in tickers.items() if v in available}
    w_arr = np.array([weights[i] for i, name in
                      enumerate(tickers.values()) if name in available])
    w_arr = w_arr / w_arr.sum()   # renormalise

    returns = compute_returns(prices[available])
    results = run_monte_carlo(returns, w_arr, config)
    plot_results(results, returns, w_arr, available_tickers, config)

    return results


if __name__ == "__main__":
    print("""
  ┌───────────────────────────────────────────────────┐
  │   MONTE CARLO PORTFOLIO RISK SIMULATOR            │
  │   Roman Falla  ·  github.com/romanfalla343-jpg    │
  │                                                   │
  │   Simulates portfolio return distributions using  │
  │   correlated GBM with Cholesky decomposition.     │
  │   Outputs: VaR, CVaR, Sharpe, Efficient Frontier  │
  └───────────────────────────────────────────────────┘
    """)
    results = run_simulator()
