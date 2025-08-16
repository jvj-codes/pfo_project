# -*- coding: utf-8 -*-
"""
Created on Thu Aug 14 10:23:03 2025

@author: nuffz
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import timedelta
from scipy.optimize import minimize
import pickle


### STEP 6.0 Load Data

# training period (2013-2025 inclusive)
train_start = "2013-01-09"
train_end   = "2025-07-16"

# Load data
print("loading data..")
file_path = r"AllReturns.xlsx"
jyske_ret = pd.read_excel(file_path, index_col=0)

# Ensure index is datetime
jyske_ret.index = pd.to_datetime(jyske_ret.index)

# Filter date range 
jyske_ret_filtered = jyske_ret.loc[train_start:train_end]

isin_ter = {'DK0016306798': 1.08, 'DK0061543600': 0.54, 'DK0060521854': 1.84, 'LU0376447149': 1.06
            , 'DK0060786564': 0.25, 'DK0060244242': 0.35, 'LU0376446257': 1.81, 'DK0016205255': 1.24
            , 'DK0060012466': 0.14, 'DK0061542719': 0.26, 'DK0060005098': 0.21, 'DK0060822468': 0.24
            , 'DK0016109614': 0.19, 'DK0016205685': 0.34, 'DK0060014678': 0.25, 'DK0061544921': 0.16
            , 'DK0016023229': 0.5, 'DK0060268506': 0.27, 'DK0061545068': 0.16, 'DK0060033975': 0.19
            , 'DK0060005254': 1.35, 'LU0320298689': 0.06, 'IE00B6R52036': 0.55, 'LU0368252358': 1.08
            , 'LU0827889139': 1.34, 'LU0252968424': 5.26, 'LU0055631609': 2.06, 'LU0724618789': 2.09
            , 'LU0788108826': 2.09, 'LU0827889303': 1.34, 'DK0060004950': 0.35, 'IE00BM67HQ30': 0.25
            , 'IE00B1FZS467': 0.65, 'LU0322253229': 0.6, 'DK0010170398': 0.85, 'DK0061544418': 0.43
            , 'DE000A0Q4R02': 0.46, 'LU0178670161': 1.07, 'LU0178670245': 1.07, 'LU0332084994': 0.8
            , 'DK0060158160': 1.44, 'DK0061553245': 0.21, 'DK0061150984': 0.22, 'DK0060227239': 0.35
            , 'DK0060118610': 0.57, 'DK0060259786': 1.34  # Benchmark
}


returns_df = jyske_ret.copy()


# --- Bootstrapping function (general purpose) ---
def bootstrap_data(train, isin, scen_len=4, num_scen=1000, seed=1230, benchmark = "DK0060259786"):
    np.random.seed(seed)
    bootstrap_results = []

    for col in isin:
        asset_returns = train[col].dropna()
        scenarios = []
        for _ in range(num_scen):
            start_idx = np.random.randint(0, len(asset_returns)-scen_len+1)
            block = asset_returns.iloc[start_idx:start_idx+scen_len]
            if len(block) == scen_len:
                scenarios.append(block.values)
        bootstrap_results.append(pd.DataFrame(scenarios, columns=[f"Week {i+1}" for i in range(scen_len)]).assign(ISIN=col))

    bootstrap_df = pd.concat(bootstrap_results, ignore_index=True)
    bootstrap_benchmark = bootstrap_df[bootstrap_df["ISIN"] == benchmark].copy()
    bootstrap_assets = bootstrap_df[bootstrap_df["ISIN"] != benchmark].copy()

    # Terminal returns
    asset_list = bootstrap_assets["ISIN"].unique()
    n_scen_per_asset = len(bootstrap_assets) // len(asset_list)
    terminal_returns = np.zeros((n_scen_per_asset, len(asset_list)))
    for j, isin in enumerate(asset_list):
        df_asset = bootstrap_df.loc[bootstrap_df["ISIN"] == isin, ["Week 1", "Week 2", "Week 3", "Week 4"]]
        cum_ret = (1 + df_asset).prod(axis=1) - 1
        terminal_returns[:, j] = cum_ret.values

    terminal_returns_bench = (1 + bootstrap_benchmark[["Week 1","Week 2","Week 3","Week 4"]]).prod(axis=1).values - 1
    terminal_returns_bench = terminal_returns_bench[:n_scen_per_asset]

    return {
        "terminal_returns": terminal_returns,
        "terminal_returns_bench": terminal_returns_bench,
        "bootstrap_assets": bootstrap_assets,
        "bootstrap_benchmark": bootstrap_benchmark,
        "asset_list": asset_list
    }

# --- Rolling bootstrap generator ---
def rolling_bootstrap(full_df, isin, init_start_date, init_end_date, scen_len=4, num_scen=1000, step=4, seed=1230):
    results = []
    returns_df_filt = full_df.loc[:, full_df.columns.isin(isin)]
    start_idx = returns_df_filt.index.get_loc(init_start_date)
    end_idx = returns_df_filt.index.get_loc(init_end_date)

    while end_idx < len(returns_df_filt):
        train_slice = returns_df_filt.iloc[start_idx:end_idx+1]
        bootstrap_window = bootstrap_data(train_slice, isin, scen_len=scen_len, num_scen=num_scen, seed=seed)

        results.append({
            "start_date": train_slice.index[0],
            "end_date": train_slice.index[-1],
            **bootstrap_window
        })

        start_idx += step
        end_idx += step

    return results

rolling_results = rolling_bootstrap(
    returns_df, 
    isin_ter,
    init_start_date="2013-01-09", 
    init_end_date="2019-01-09", 
    scen_len=4, 
    num_scen=1000, 
    step=4
)

for r in rolling_results:
    print(f"{r['start_date']} → {r['end_date']}, scenarios: {r['terminal_returns'].shape}")
    
# ---------------------------
# Strategy 1: Minimize portfolio STD s.t. expected return >= benchmark_mean
# ---------------------------
# --- Portfolio functions ---
def portfolio_return(weights, mu):
    return float(weights @ mu)

def portfolio_variance(weights, cov):
    return float(weights @ cov.values @ weights)

def portfolio_std(weights, cov):
    return np.sqrt(portfolio_variance(weights, cov))

def optimize_portfolio(mu, cov, constraints=[], w_prev=None, bounds=None, cost_rate=0.001):
    n = len(mu)
    
    def obj_fun(w):
        var = portfolio_variance(w, cov)
        tc = cost_rate * np.sum(np.abs(w - w_prev)) if w_prev is not None else 0.0
        return var + tc

    x0 = w_prev if w_prev is not None else np.ones(n)/n

    res = minimize(obj_fun, x0, method="SLSQP", bounds=bounds, constraints=constraints,
                   options={"ftol": 1e-9, "maxiter":1000})
    return res

def min_std_benchmark_strategy(rolling_bootstrap_results, cost_rate=0.001):
    strategy_results = []

    w_prev = None  # initial portfolio is cash or equal weights
    
    for window in rolling_bootstrap_results:
        terminal_returns = window["terminal_returns"]
        terminal_returns_bench = window["terminal_returns_bench"]
        assets = window["asset_list"]
        n_assets = len(assets)

        # Compute mean and covariance from bootstrapped terminal returns
        mu = pd.Series(terminal_returns.mean(axis=0), index=assets)
        cov = pd.DataFrame(np.cov(terminal_returns.T), index=assets, columns=assets)

        # Benchmark target: mean of bootstrapped benchmark
        bench_target = terminal_returns_bench.mean()

        # Constraints
        cons = [
            {"type": "eq", "fun": lambda w: np.sum(w) - 1.0},  # sum(weights) == 1
            {"type": "ineq", "fun": lambda w, mu=mu.values, target=bench_target: (w @ mu) - target}
        ]

        bounds = [(0.0, 1.0)] * n_assets

        # Run optimization using previous weights to include transaction cost
        res = optimize_portfolio(mu, cov, constraints=cons, w_prev=w_prev, bounds=bounds, cost_rate=cost_rate)

        if res.success:
            w_opt = res.x
            port_return = portfolio_return(w_opt, mu.values)
            port_std = portfolio_std(w_opt, cov)
            strategy_results.append({
                "start_date": window["start_date"],
                "end_date": window["end_date"],
                "weights": w_opt,
                "expected_return": port_return,
                "expected_std": port_std
            })
            w_prev = w_opt  # update previous portfolio for next window
        else:
            print(f"Optimization failed for window {window['start_date']} - {window['end_date']}: {res.message}")
            strategy_results.append({
                "start_date": window["start_date"],
                "end_date": window["end_date"],
                "weights": None,
                "expected_return": None,
                "expected_std": None
            })
    
    return strategy_results

strategy_results = min_std_benchmark_strategy(
    rolling_bootstrap_results=rolling_results,
    cost_rate=0.001  # 0.1% transaction cost per buy/sell
)



assets = rolling_results[0]["asset_list"]

# dates as index, assets as columns
weights_df = pd.DataFrame([
    dict(date=sr["end_date"], **dict(zip(assets, sr["weights"])))
    for sr in strategy_results if sr["weights"] is not None
])

weights_df.set_index("date", inplace=True)

def get_distinct_colors(n):
    base_maps = ['tab20', 'Set3', 'tab10', 'Paired', 'Accent']
    colors = []
    for cmap_name in base_maps:
        cmap = plt.get_cmap(cmap_name)
        num_colors = cmap.N  
        for i in range(num_colors):
            colors.append(cmap(i))
    # Ensure we have enough and slice to n
    return colors[:n]


color_list = get_distinct_colors(45)

weights_df.plot.area(
    colormap=plt.matplotlib.colors.ListedColormap(color_list),
    alpha=0.85,
    linewidth=0
)

plt.ylabel("Portfolio Weight")
plt.title("Portfolio Composition Over Time")
plt.legend(fontsize=6, ncol=5, loc="upper center", bbox_to_anchor=(0.5, -0.2))
plt.tight_layout()
plt.show()


# Ex post plot ----------------------------------
# Load CVaR results
with open(r"strategy_results_cvar.pkl", "rb") as f:
    loaded_results_cvar = pickle.load(f) 


# Initialize
port_returns = []
port_returns2 = []
portfolio_values = [100.0]
portfolio_values2 = [100.0] 
mean_values = [100.0]
best_values = [100.0]
worst_values2 = [100.0]
mean_values2 = [100.0]
best_values2 = [100.0]
worst_values = [100.0]
agg_mean_values = [100.0]
agg_best_values = [100.0]
agg_worst_values = [100.0]

dates = []

# Loop
for roll, strat, strat_cvar in zip(rolling_results, strategy_results, loaded_results_cvar):
    if strat["weights"] is None or strat_cvar["weights"] is None:
        continue

    w = pd.Series(strat["weights"], index=roll["asset_list"])
    w_cvar = pd.Series(strat_cvar["weights"], index=roll["asset_list"])

    start = pd.to_datetime(roll["end_date"]) + pd.Timedelta(days=1)
    end = start + pd.Timedelta(weeks=4) - pd.Timedelta(days=1)

    period_returns = returns_df.loc[start:end, roll["asset_list"]].copy()
    if period_returns.shape[0] == 0:
        dates.append(pd.to_datetime(roll["end_date"]))
        portfolio_values.append(portfolio_values[-1])
        portfolio_values2.append(portfolio_values2[-1])
        mean_values.append(mean_values[-1])
        best_values.append(best_values[-1])
        worst_values.append(worst_values[-1])
        continue

    # Portfolio 1
    port_period_returns = (period_returns @ w)
    port_returns.append(port_period_returns)
    growth_factor = (1.0 + port_period_returns).prod()
    new_value = portfolio_values[-1] * growth_factor

    # Portfolio 2 (CVaR)
    port_period_returns_cvar = (period_returns @ w_cvar)
    port_returns2.append(port_period_returns_cvar)
    growth_factor_cvar = (1.0 + port_period_returns_cvar).prod()
    new_value_cvar = portfolio_values2[-1] * growth_factor_cvar

    # Ex-ante for Portfolio 1
    terminal_returns = roll["terminal_returns"]
    scen_returns = terminal_returns @ strat["weights"]
    mean_ret = float(np.mean(scen_returns))
    best_ret = float(np.max(scen_returns))
    worst_ret = float(np.min(scen_returns))
    # Ex-ante for Portfolio 2
    terminal_returns = roll["terminal_returns"]
    scen_returns = terminal_returns @ strat_cvar["weights"]
    mean_ret2 = float(np.mean(scen_returns))
    best_ret2 = float(np.max(scen_returns))
    worst_ret2 = float(np.min(scen_returns))

    # Append
    
    dates.append(pd.to_datetime(roll["end_date"]))
    portfolio_values.append(new_value)
    portfolio_values2.append(new_value_cvar)
    agg_mean_values.append(mean_values[-1] * (1.0 + mean_ret))
    agg_best_values.append(best_values[-1] * (1.0 + best_ret))
    agg_worst_values.append(worst_values[-1] * (1.0 + worst_ret))
    mean_values.append(portfolio_values[-1] * (1.0 + mean_ret))
    best_values.append(portfolio_values[-1] * (1.0 + best_ret))
    worst_values.append(portfolio_values[-1] * (1.0 + worst_ret))
    mean_values2.append(portfolio_values2[-1] * (1.0 + mean_ret))
    best_values2.append(portfolio_values2[-1] * (1.0 + best_ret))
    worst_values2.append(portfolio_values2[-1] * (1.0 + worst_ret))

# Convert to Series
dates_idx = pd.DatetimeIndex(pd.to_datetime(dates))
delta = dates_idx[-1] - dates_idx[-2]
new_date = dates_idx[-1] + delta

extended_dates = dates_idx.append(pd.DatetimeIndex([new_date]))

portfolio_series = pd.Series(portfolio_values, index=extended_dates)
portfolio_series2 = pd.Series(portfolio_values2, index=extended_dates)
mean_series = pd.Series(mean_values, index=extended_dates)
best_series = pd.Series(best_values, index=extended_dates)
worst_series = pd.Series(worst_values, index=extended_dates)
mean_series2 = pd.Series(mean_values2, index=extended_dates)
best_series2 = pd.Series(best_values2, index=extended_dates)
worst_series2 = pd.Series(worst_values2, index=extended_dates)

# Plot with Portfolio 2
plt.figure(figsize=(12,6))
plt.plot(portfolio_series.index, portfolio_series.values, label="Actual Portfolio Value (ex-post)", color="black", lw=2)
plt.plot(mean_series.index, mean_series.values, label="Ex-ante Mean", color="blue", ls="--")
plt.plot(best_series.index, best_series.values, label="Ex-ante Best Case", color="green", ls=":")
plt.plot(worst_series.index, worst_series.values, label="Ex-ante Worst Case", color="red", ls=":")
plt.fill_between(mean_series.index, worst_series.values, best_series.values, color="gray", alpha=0.2, label="Ex-ante Range")

plt.ylabel("Portfolio Value")
plt.title("Markowitz Portfolio Value with Ex-Ante Scenario Bounds")
plt.legend()
plt.tight_layout()
plt.show()


plt.figure(figsize=(12,6))
plt.plot(portfolio_series2.index, portfolio_series2.values, label="CVaR Portfolio Value", color="purple", lw=2)
plt.plot(mean_series2.index, mean_series2.values, label="Ex-ante Mean", color="blue", ls="--")
plt.plot(best_series2.index, best_series2.values, label="Ex-ante Best Case", color="green", ls=":")
plt.plot(worst_series2.index, worst_series2.values, label="Ex-ante Worst Case", color="red", ls=":")
plt.fill_between(mean_series2.index, worst_series2.values, best_series2.values, color="gray", alpha=0.2, label="Ex-ante Range")

plt.ylabel("Portfolio Value")
plt.title("CVaR Portfolio Value with Ex-Ante Scenario Bounds")
plt.legend()
plt.tight_layout()
plt.show()

# Benchmark plot also with Portfolio 2
benchmark_isin = "DK0060259786"
benchmark_values = [100.0]
bench_returns = []
for roll in rolling_results:
    start = pd.to_datetime(roll["end_date"]) + pd.Timedelta(days=1)
    end = start + pd.Timedelta(weeks=4) - pd.Timedelta(days=1)
    bench_period = returns_df.loc[start:end, benchmark_isin]
    if bench_period.shape[0] == 0:
        benchmark_values.append(benchmark_values[-1])
    else:
        growth_factor = (1.0 + bench_period).prod()
        benchmark_values.append(benchmark_values[-1] * growth_factor)
        bench_returns.append(growth_factor)

benchmark_series = pd.Series(benchmark_values, index=extended_dates)

plt.figure(figsize=(12,6))
plt.plot(portfolio_series.index, portfolio_series.values, color="black", lw=2, label="Markowitz Portfolio Value")
plt.plot(portfolio_series2.index, portfolio_series2.values, color="purple", lw=2, label="CVaR Portfolio Value")
plt.plot(benchmark_series.index, benchmark_series.values, color="orange", lw=2, label="Benchmark Value")
plt.ylabel("Value")
plt.title("Markowitz Portfolio vs CVaR Portfolio vs Benchmark")
plt.legend()
plt.tight_layout()
plt.show()


mapping_df = pd.read_excel(r"NameISIN.xlsx")  
isin_to_name = dict(zip(mapping_df["ISIN"], mapping_df["Name"]))

# Rename columns in weights_df
weights_named_df = weights_df.rename(columns=isin_to_name)

# Now plot using the named columns
assets = weights_named_df.columns
dates = weights_named_df.index
color_list = get_distinct_colors(len(assets))

fig, ax = plt.subplots(figsize=(12, 6))

bottom = np.zeros(len(dates))
for asset, color in zip(assets, color_list):
    ax.bar(dates, weights_named_df[asset], bottom=bottom, color=color, width=20, label=asset)
    bottom += weights_named_df[asset].values

ax.set_ylabel("Portfolio Weight")
ax.set_title("Markowitz Portfolio Composition per Period")
ax.set_ylim(0, 1.0)
ax.legend(fontsize=6, ncol=5, loc="upper center", bbox_to_anchor=(0.5, -0.15))
plt.tight_layout()
plt.show()


# Plot 2
weights_df2 = pd.DataFrame([
    dict(date=sr["end_date"], **dict(zip(assets, sr["weights"])))
    for sr in loaded_results_cvar if sr["weights"] is not None
])

weights_df2.set_index("date", inplace=True)
weights_named_df2 = weights_df2.rename(columns=isin_to_name)

assets2 = weights_named_df2.columns
dates2 = weights_named_df2.index
threshold = 1e-4
color_list2 = get_distinct_colors(len(assets2))

fig, ax = plt.subplots(figsize=(12, 6))

bottom = np.zeros(len(dates2))
for asset, color in zip(assets2, color_list2):
    ax.bar(dates2, weights_named_df2[asset], bottom=bottom, color=color, width=20, label=asset)
    bottom += weights_named_df2[asset].values

ax.set_ylabel("Portfolio Weight")
ax.set_title("CVaR Portfolio Composition per Period")
ax.set_ylim(0, 1.0)
ax.legend(fontsize=6, ncol=5, loc="upper center", bbox_to_anchor=(0.5, -0.15))
plt.tight_layout()
plt.show()

def portfolio_stats(portfolio_values, periods_per_year=13):
    returns = portfolio_values.pct_change().dropna()
    
    if isinstance(returns, pd.Series):
        returns = returns.to_frame('Portfolio')
    
    results = {}
    for col in returns.columns:
        r = returns[col]
        mean_ann = ((1 + r).prod())**(periods_per_year / len(r)) - 1  
        std_ann = r.std() * np.sqrt(periods_per_year)
        var_99 = np.percentile(r, 1) # alpha = 0.99
        cvar_99 = r[r <= var_99].mean()
        results[col] = {
            'Mean Annual Return (%)': mean_ann * 100,
            'Annualized Std (%)': std_ann * 100,
            'VaR 0.99 (%)': var_99 * 100,
            'CVaR 0.99 (%)': cvar_99 * 100
        }
    
    return pd.DataFrame(results).T

markowitz = portfolio_stats(portfolio_series)
cvar = portfolio_stats(portfolio_series2)
bench = portfolio_stats(benchmark_series)

# Now annualize
def annualized_exp_return(returns, periods_per_year):
    """Calculate annualized return from periodic returns"""
    compounded_growth = (1 + returns).prod()
    n_periods = returns.shape[0]
    return compounded_growth**(periods_per_year / n_periods) - 1

def annualized_return(returns):
    """Calculate annualized return"""
    annualized_ret = (1 + returns).resample('YE').prod() -1
    return annualized_ret

# VaR at 95%
def value_at_risk(returns, alpha=0.05):
    return returns.quantile(alpha)

# Keep your existing functions:
def calculate_cvar(returns, alpha=0.05):
    if returns.empty:
        return np.nan
    var = returns.quantile(alpha)
    return returns[returns <= var].mean()

def sharpe_ratio(ann_returns, ann_std, risk_free_rate=0.0225):
    excess_returns = ann_returns - risk_free_rate
    return excess_returns.mean() / ann_std

periods_per_year = 13  

summary_stats = {}
portfolio_series_ret = portfolio_series.pct_change().iloc[1:]
portfolio_series2_ret = portfolio_series2.pct_change().iloc[1:]
benchmark_series_ret = benchmark_series.pct_change().iloc[1:]

for name, ret_series in zip(
    ['Min CVaR', 'Max Return w/ CVaR ≤ Benchmark', 'Benchmark'],
    [portfolio_series_ret, portfolio_series2_ret, benchmark_series_ret]
):
    mean = ret_series.mean() * periods_per_year
    std = ret_series.std() * np.sqrt(periods_per_year)
    ann_exp_ret = annualized_exp_return(ret_series, periods_per_year)
    ann_ret = annualized_return(ret_series)
    sharpe = sharpe_ratio(ann_ret, std, risk_free_rate = 0.0225)
    var95 = abs(value_at_risk(ann_ret, alpha=0.05))
    cvar95 = abs(calculate_cvar(ann_ret, alpha=0.05))
    
    summary_stats[name] = {
        'Mean Annualized (%)': ann_exp_ret*100,
        'STD Annualized (%)': std*100,
        'Sharpe Ratio': sharpe,
        'VaR(95%) (%)': var95*100,
        'CVaR(95%) (%)': cvar95*100
    }

pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_rows', None)
summary_df = pd.DataFrame(summary_stats).T.round(2)
print(summary_df)
