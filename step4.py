# -*- coding: utf-8 -*-
"""
Created on Sat Aug  9 16:43:10 2025

@author: nuffz
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import timedelta
from scipy.optimize import minimize


# training period (2013-2019 inclusive)
train_start = "2013-01-09"
train_end   = "2019-01-09"

# Load data

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


jyske_ret_filtered = jyske_ret_filtered.loc[:, jyske_ret_filtered.columns.isin(isin_ter)]

returns = jyske_ret_filtered.copy()  
train = returns.loc[train_start:train_end].dropna(how="all", axis=1)  

# Calculate periods per year

deltas = np.diff(train.index.values).astype('timedelta64[D]').astype(int)
median_days = np.median(deltas)
periods_per_year = int(round(365.25 / median_days)) # gives exactly 52

print(periods_per_year, "\n")

mu_sample = train.mean(axis=0)       
cov_sample = train.cov()             

mu_ann = (1 + mu_sample).pow(periods_per_year) - 1
cov_ann = cov_sample * periods_per_year

assets = mu_ann.index.tolist()
n = len(assets)

benchmark = "DK0060259786" # Balanceret
bench_mean = mu_ann.loc[benchmark]
bench_std = np.sqrt(cov_ann.loc[benchmark, benchmark])

mu_ann_assets = mu_ann.drop(index=benchmark)
cov_ann_assets = cov_ann.drop(index=benchmark, columns=benchmark)

# new asset list and dimensions (without benchmark)
assets = mu_ann_assets.index.tolist()
mu_ann = mu_ann_assets
cov_ann = cov_ann_assets
n = len(assets)

def portfolio_return(weights, mu):
    return float(weights @ mu)

def portfolio_variance(weights, cov):
    return float(weights @ cov.values @ weights)

def portfolio_std(weights, cov):
    return np.sqrt(portfolio_variance(weights, cov))

bounds = [(0.0, 1.0)] * n        # no shorting, no leverage

cons = [
    {"type": "eq", "fun": lambda w: np.sum(w) - 1.0}  # sum(weights) == 1
]

# 1) Maximize expected return s.t. portfolio STD <= benchmark_std ----
# convert to minimization
def neg_ret_obj(w, mu):
    return - portfolio_return(w, mu)

def std_ineq(w, cov, max_std):
    return max_std**2 - portfolio_variance(w, cov)  # require variance <= max_std^2

cons1 = cons + [
    {"type": "ineq", "fun": lambda w, cov=cov_ann, mx=bench_std: std_ineq(w, cov, mx)}
]

x0 = np.repeat(1.0/n, n)

res1 = minimize(neg_ret_obj, x0, args=(mu_ann.values,), method="SLSQP",
                bounds=bounds, constraints=cons1, options={"ftol":1e-9, "maxiter":1000})

if not res1.success:
    print("Problem 1 solver warning/failure:", res1.message)

w1 = res1.x
r1 = portfolio_return(w1, mu_ann.values)
s1 = portfolio_std(w1, cov_ann)
print("\n--- Problem 1 (max return subject to STD <= benchmark STD) ---")
print("Success:", res1.success)
print("Objective (annual return):", r1)
print("Portfolio STD:", s1)
print("Weights (nonzero):")
print(pd.Series(w1, index=assets).loc[lambda s: s.abs() > 1e-6].sort_values(ascending=False))

# 2) Minimize portfolio STD s.t. expected return >= benchmark_mean ----

def var_obj(w, cov):
    return portfolio_variance(w, cov)

cons2 = cons + [
    {"type": "ineq", "fun": lambda w, mu=mu_ann, target=bench_mean: (w @ mu) - target}
]

res2 = minimize(var_obj, x0, args=(cov_ann,), method="SLSQP",
                bounds=bounds, constraints=cons2, options={"ftol":1e-9, "maxiter":1000})

if not res2.success:
    print("Problem 2 solver warning/failure:", res2.message)

w2 = res2.x
r2 = portfolio_return(w2, mu_ann.values)
s2 = portfolio_std(w2, cov_ann)
print("\n--- Problem 2 (min STD subject to return >= benchmark mean) ---")
print("Success:", res2.success)
print("Objective (annual return):", r2)
print("Portfolio STD:", s2)
print("Weights (nonzero):")
print(pd.Series(w2, index=assets).loc[lambda s: s.abs() > 1e-6].sort_values(ascending=False))


# Save results
summary = pd.DataFrame({
    "weight_max_return": w1,
    "weight_min_std": w2,
    "mu_ann": mu_ann,
    "sigma_ann": np.sqrt(np.diag(cov_ann))
}, index=assets)

summary.to_excel("markowitz_results_2013_2019.xlsx")
print("\nSaved results to markowitz_results_2013_2019.xlsx")

# Now to 4.2 -----------------------------------------
def var_obj(w, cov):
    return portfolio_variance(w, cov)

def obj_def(w, cov, mu, lbd):
    return -((1-lbd)*portfolio_return(w, mu) - lbd*portfolio_variance(w, cov))

cons_eq = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
x0_mvp = np.repeat(1.0/n, n)

lambda_min = 0.999
res_mvp_min = minimize(obj_def, x0_mvp, args=(cov_ann,mu_ann,lambda_min), method="SLSQP",
                   bounds=bounds, constraints=cons_eq, options={"ftol":1e-9, "maxiter":1000})

w_mvp_min = res_mvp_min.x
min_var = portfolio_variance(w_mvp_min, cov_ann)
min_std = np.sqrt(min_var)

# upper variance bound: max single-asset variance 
lambda_max = 0.000
res_mvp_max = minimize(obj_def, x0_mvp, args=(cov_ann,mu_ann,lambda_max), method="SLSQP",
                   bounds=bounds, constraints=cons_eq, options={"ftol":1e-9, "maxiter":1000})
w_mvp_max = res_mvp_max.x
max_var = portfolio_variance(w_mvp_max, cov_ann)
max_std = np.sqrt(max_var)

# 10 evenly spaced variance levels
num_points = 10
#var_grid = (np.linspace(min_var, max_var,) num_points)
StdStepLength = (max_std - min_std) / (num_points - 1)

frontier_weights = []
frontier_rets = []
frontier_stds = []

# First point: min variance portfolio
frontier_weights.append(w_mvp_min)
frontier_rets.append(portfolio_return(w_mvp_min, mu_ann.values))
frontier_stds.append(min_std)

x0 = w_mvp_min.copy()

# Intermediate points
for i in range(1, num_points - 1):
    StdLim = min_std + i * StdStepLength

    cons_target = cons_eq + [
        {"type": "ineq",
         "fun": lambda w, cov=cov_ann, mx=StdLim: mx - portfolio_std(w, cov)}
    ]

    res = minimize(neg_ret_obj, x0, args=(mu_ann.values,),
                   method="SLSQP", bounds=bounds, constraints=cons_target,
                   options={"ftol": 1e-9, "maxiter": 1000})

    if res.success:
        w = res.x
        r = portfolio_return(w, mu_ann.values)
        s = portfolio_std(w, cov_ann)
        frontier_weights.append(w)
        frontier_rets.append(r)
        frontier_stds.append(s)
        x0 = w  # warm start
    else:
        print(f"Could not find solution for target Std {StdLim:.6f}")

# std and returns for efficient frontier
frontier_stds = np.array(frontier_stds)
frontier_rets = np.array(frontier_rets)

# plot
plt.figure(figsize=(9,6))

plt.plot(frontier_stds, frontier_rets, marker='o', label='Efficient frontier (10 pts)')

asset_stds = np.sqrt(np.diag(cov_ann))
asset_rets = mu_ann.values
plt.scatter(asset_stds, asset_rets, marker='x', label='Assets')
for i, a in enumerate(assets):
    plt.annotate(a, (asset_stds[i], asset_rets[i]), xytext=(4,0), textcoords='offset points', fontsize=8)

plt.scatter([s1], [r1], marker='D', s=80, label='Max-return @ bench STD', zorder=5)
plt.scatter([s2], [r2], marker='s', s=80, label='Min-STD @ bench return', zorder=5)

plt.scatter([bench_std], [bench_mean], marker='*', s=140, label='Benchmark', color='gold', edgecolor='k', zorder=6)
#mvp_r = portfolio_return(w_mvp, mu_ann.values)
#plt.scatter([min_std], [mvp_r], marker='P', s=80, label='Global min-variance (MVP)', zorder=5)

plt.xlabel('Portfolio standard deviation (annual)')
plt.ylabel('Portfolio expected return (annual)')
plt.title('Efficient frontier')
plt.grid(True)
plt.legend(loc='best')
plt.tight_layout()

plt.savefig("efficient_frontier_2013_2019.png", dpi=300)
plt.show()
print("Saved as efficient_frontier_2013_2019.png")

################ Cumulative Return
# Out-of-sample period for performance evaluation

perf_start = '2019-01-16'
perf_end   = '2025-07-23'

# Filter returns for performance period, keep only assets in portfolio + benchmark
portfolio_assets = [i for i in isin_ter.keys() if i != benchmark]

perf_start = pd.to_datetime("2019-01-16")
perf_end = pd.to_datetime("2025-12-31")

returns_perf = jyske_ret.loc[(jyske_ret.index >= perf_start) & (jyske_ret.index <= perf_end), portfolio_assets]
returns_perf_bm = jyske_ret.loc[(jyske_ret.index >= perf_start) & (jyske_ret.index <= perf_end), [benchmark]]



# Portfolio returns
port1_returns = returns_perf.dot(w1).fillna(0)
port2_returns = returns_perf.dot(w2).fillna(0)
bench_returns = returns_perf_bm.squeeze().fillna(0)  # convert to Series

# Cumulative returns
# Cumulative returns, rebased to start at 100
cum_ret1 = ((1 + port1_returns).cumprod()) * 100 / ((1 + port1_returns).cumprod().iloc[0])
cum_ret2 = ((1 + port2_returns).cumprod()) * 100 / ((1 + port2_returns).cumprod().iloc[0])
cum_bench = ((1 + bench_returns).cumprod()) * 100 / ((1 + bench_returns).cumprod().iloc[0])


# Plot cumulative returns
plt.figure(figsize=(10,6))
plt.plot(cum_ret1.index, cum_ret1, label="Max Return Portfolio")
plt.plot(cum_ret2.index, cum_ret2, label="Min Variance Portfolio")
plt.plot(cum_bench.index, cum_bench, label="Benchmark (Balanced)", linestyle='--', color='black')

plt.title("Value Growth 2019-2025")
plt.xlabel("Date")
plt.ylabel("Portfolio Value (Indexed to 100)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

### CALCULATE PERFORMANCE OF THE STRAGIES

# Annualized expected return
def annualized_return(returns, periods_per_year):
    """Calculate annualized return from periodic returns"""
    compounded_growth = (1 + returns).prod()
    n_periods = returns.shape[0]
    return compounded_growth**(periods_per_year / n_periods) - 1

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


periods_per_year = 52  

summary_stats = {}

for name, ret_series in zip(
    ['Max Return', 'Min Variance', 'Benchmark'],
    [port1_returns, port2_returns, bench_returns]
):
    mean = ret_series.mean() * periods_per_year
    std = ret_series.std() * np.sqrt(periods_per_year)
    ann_ret = annualized_return(ret_series, periods_per_year)
    sharpe = sharpe_ratio(ann_ret, std, risk_free_rate = 0.0225)
    var95 = value_at_risk(ret_series, alpha=0.05)
    cvar95 = calculate_cvar(ret_series, alpha=0.05)
    
    summary_stats[name] = {
        'Mean Annualized (%)': ann_ret*100,
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

