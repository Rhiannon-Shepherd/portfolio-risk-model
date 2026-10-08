"""
Multi-asset portfolio risk model.

Measures performance and correlations for an 8-asset ETF portfolio,
estimates 1-day VaR and Expected Shortfall three ways, and backtests
rolling VaR with a Kupiec test. Results and discussion are in README.md.

Requires: pip install yfinance pandas numpy scipy matplotlib
"""

import os
import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

pd.set_option("display.float_format", "{:.4f}".format)
plt.rcParams["figure.figsize"] = (10, 4)
os.makedirs("figures", exist_ok=True)

# %% 1. Data

# SPY - US large-cap equities (S&P 500)
# EFA - developed markets ex-US equities
# EEM - emerging-market equities
# TLT - long-dated US Treasuries
# IEF - 7-10 year US Treasuries
# LQD - investment-grade corporate bonds
# GLD - gold
# UUP - US dollar index
TICKERS = ["SPY", "EFA", "EEM", "TLT", "IEF", "LQD", "GLD", "UUP"]
START, END = "2015-01-01", "2026-09-30"


def load_prices(tickers, start, end, cache="prices.csv"):
    """Download adjusted prices once, then reuse the saved CSV."""
    try:
        return pd.read_csv(cache, index_col=0, parse_dates=True)
    except FileNotFoundError:
        px = yf.download(tickers, start=start, end=end,
                         auto_adjust=True, progress=False)["Close"]
        px.to_csv(cache)
        return px


prices = load_prices(TICKERS, START, END)
prices = prices[TICKERS].dropna()   # keep only days where every asset traded
print(prices.shape, prices.index.min().date(), "to", prices.index.max().date())
print(prices.tail())

# %%

(prices / prices.iloc[0] * 100).plot(title="Prices rebased to 100", logy=True)
plt.savefig("figures/prices_rebased.png", dpi=150, bbox_inches="tight")
plt.show()

# %% 2. Returns and performance

# simple returns, so portfolio = weighted sum
rets = prices.pct_change().dropna()
TD = 252


def load_rf(start, end, cache="irx.csv"):
    """13-week US T-bill yield (in %), downloaded once then reused."""
    try:
        return pd.read_csv(cache, index_col=0, parse_dates=True).squeeze("columns")
    except FileNotFoundError:
        irx = yf.download("^IRX", start=start, end=end,
                          progress=False)["Close"].squeeze()
        irx.to_csv(cache)
        return irx


irx = load_rf(START, END)
rf_annual = irx.reindex(rets.index).ffill().mean() / 100
print(f"Average risk-free rate: {rf_annual:.2%}")


def perf_table(r, rf):
    ann_ret = (1 + r).prod() ** (TD / len(r)) - 1
    ann_vol = r.std() * np.sqrt(TD)
    sharpe = (ann_ret - rf) / ann_vol
    wealth = (1 + r).cumprod()
    max_dd = (wealth / wealth.cummax() - 1).min()
    return pd.DataFrame({"Ann. return": ann_ret, "Ann. vol": ann_vol,
                         "Sharpe": sharpe, "Max drawdown": max_dd})


# equal weights, rebalanced daily
w = np.repeat(1 / len(TICKERS), len(TICKERS))
rets["Portfolio"] = rets[TICKERS] @ w

print(perf_table(rets, rf=rf_annual))

# %%

corr = rets[TICKERS].corr()
fig, ax = plt.subplots(figsize=(7, 6))
im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
ax.set_xticks(range(len(TICKERS)), TICKERS)
ax.set_yticks(range(len(TICKERS)), TICKERS)
for i in range(len(TICKERS)):
    for j in range(len(TICKERS)):
        ax.text(j, i, f"{corr.iloc[i, j]:.2f}",
                ha="center", va="center", fontsize=8)
fig.colorbar(im)
ax.set_title("Correlation of daily returns")
plt.savefig("figures/correlation_heatmap.png", dpi=150, bbox_inches="tight")
plt.show()

# %% 3. VaR and Expected Shortfall
# losses reported as positive numbers


def var_es_historical(r, alpha):
    q = np.quantile(r, 1 - alpha)
    return -q, -r[r <= q].mean()


def var_es_parametric(r, alpha):
    mu, sd = r.mean(), r.std()
    z = stats.norm.ppf(1 - alpha)
    var = -(mu + z * sd)
    es = -(mu - sd * stats.norm.pdf(z) / (1 - alpha))
    return var, es


def var_es_montecarlo(asset_rets, weights, alpha, n=100_000, seed=42):
    # multivariate normal, so should roughly match parametric
    rng = np.random.default_rng(seed)
    sims = rng.multivariate_normal(asset_rets.mean(), asset_rets.cov(), size=n)
    port = sims @ weights
    return var_es_historical(port, alpha)


port = rets["Portfolio"]
rows = []
for a in [0.95, 0.99]:
    for name, (v, e) in {
        "Historical": var_es_historical(port, a),
        "Parametric": var_es_parametric(port, a),
        "Monte Carlo": var_es_montecarlo(rets[TICKERS], w, a),
    }.items():
        rows.append({"Confidence": f"{a:.0%}",
                    "Method": name, "VaR": v, "ES": e})
var_table = pd.DataFrame(rows).set_index(["Confidence", "Method"])
print((var_table * 100).round(2).rename(columns=lambda c: c + " (%)"))

# bootstrap CI for 99% historical VaR
rng = np.random.default_rng(0)
boot = []
for _ in range(1000):
    sample = port.sample(len(port), replace=True,
                         random_state=rng.integers(1e9))
    boot.append(var_es_historical(sample, 0.99)[0])
lo, hi = np.percentile(boot, [2.5, 97.5])
print(f"99% historical VaR 95% confidence interval: {lo:.2%} to {hi:.2%}")

# %%

print(
    f"Skew: {stats.skew(port):.2f}   Excess kurtosis: {stats.kurtosis(port):.2f}")

ax = port.hist(bins=100, density=True, alpha=0.6, label="Actual")
x = np.linspace(port.min(), port.max(), 400)
ax.plot(x, stats.norm.pdf(x, port.mean(), port.std()), label="Normal fit")
ax.set_title("Portfolio daily returns vs normal distribution")
ax.legend()
plt.savefig("figures/returns_vs_normal.png", dpi=150, bbox_inches="tight")
plt.show()

# %% 4. Backtesting VaR
# rolling 250-day VaR, checked against next-day loss
# Kupiec POF test: is the breach rate consistent with 1%? (p < 0.05 = reject)

WINDOW, ALPHA = 250, 0.99


def rolling_var(r, window, alpha, method):
    out = pd.Series(index=r.index, dtype=float)
    for t in range(window, len(r)):
        past = r.iloc[t - window:t]   # excludes day t, no look-ahead
        if method == "historical":
            out.iloc[t] = -np.quantile(past, 1 - alpha)
        else:  # parametric normal
            out.iloc[t] = - \
                (past.mean() + stats.norm.ppf(1 - alpha) * past.std())
    return out.dropna()


def kupiec(breaches, alpha):
    n, x = len(breaches), int(breaches.sum())
    p = 1 - alpha
    p_hat = x / n
    if x == 0:
        lr = -2 * n * np.log(1 - p)
    else:
        lr = -2 * ((n - x) * np.log(1 - p) + x * np.log(p)
                   - (n - x) * np.log(1 - p_hat) - x * np.log(p_hat))
    return {"Days": n, "Breaches": x, "Expected": round(n * p, 1),
            "Breach rate": p_hat, "LR stat": lr, "p-value": 1 - stats.chi2.cdf(lr, 1)}


results, forecasts = {}, {}
for m in ["historical", "parametric"]:
    v = rolling_var(port, WINDOW, ALPHA, m)
    forecasts[m] = v
    breaches = (-port.loc[v.index]) > v
    results[m] = kupiec(breaches, ALPHA)
print(pd.DataFrame(results).T.astype({"Days": int, "Breaches": int}))

# %%

v = forecasts["parametric"]
loss = -port.loc[v.index]
hit = loss > v
plt.plot(loss.index, loss, lw=0.5, color="grey", label="Daily loss")
plt.plot(v.index, v, color="C0", label="99% parametric VaR")
plt.scatter(loss.index[hit], loss[hit], color="red",
            s=12, label="Breach", zorder=3)
plt.title("Rolling 99% VaR vs realised losses")
plt.legend()
plt.savefig("figures/rolling_var_backtest.png", dpi=150, bbox_inches="tight")
plt.show()

# breaches per year, both models
yearly = {}
for m, v in forecasts.items():
    hits = (-port.loc[v.index]) > v
    yearly[f"{m.capitalize()} breaches"] = hits.groupby(hits.index.year).sum()

breach_table = pd.DataFrame(yearly)
breach_table.index.name = "Year"
breach_table["Expected"] = (hits.groupby(
    hits.index.year).size() * (1 - ALPHA)).round(1)
breach_table.loc["Total"] = breach_table.sum()
breach_table = breach_table.astype({"Historical breaches": int,
                                    "Parametric breaches": int})
print(breach_table.to_string(float_format="{:.1f}".format))
