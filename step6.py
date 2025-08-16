import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import timedelta
from scipy.optimize import minimize

### STEP 6.0 Load Data

# training period (2013-2019 inclusive)
train_start = "2013-01-09"
train_end   = "2019-01-09"

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

jyske_ret_filtered = jyske_ret_filtered.loc[:, jyske_ret_filtered.columns.isin(isin_ter.keys())]

returns = jyske_ret_filtered.copy()  
train = returns.loc[train_start:train_end].dropna(how="all", axis=1)  
# Some calculations - perhaps necessary 
deltas = np.diff(train.index.values).astype('timedelta64[D]').astype(int)
median_days = np.median(deltas)
periods_per_year = int(round(365.25 / median_days)) # gives exactly 52

#print(periods_per_year, "\n")

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

### STEP 6.1 -------
print("bootstrapping ..")
np.random.seed(1230)
n_scenarios = 1000
scenario_length = 4  

bootstrap_results = []

for isin in train.columns:
    asset_returns = train[isin].dropna() 
    dates = asset_returns.index
    
    scenarios = []
    for _ in range(n_scenarios):
        start_idx = np.random.randint(0, len(asset_returns) - scenario_length + 1)
        
        block = asset_returns.iloc[start_idx:start_idx + scenario_length]
        
        if len(block) == scenario_length:
            scenarios.append(block.values)
    
    bootstrap_results.append(pd.DataFrame(scenarios, columns=[f"Week {i+1}" for i in range(scenario_length)]).assign(ISIN=isin))

bootstrap_df = pd.concat(bootstrap_results, ignore_index=True)
bootstrap_df_full = bootstrap_df

### STEP 6.2 -------
## Fjerner benchmark
bootstrap_benchmark = bootstrap_df[bootstrap_df["ISIN"] == benchmark].copy()
bootstrap_assets = bootstrap_df[bootstrap_df["ISIN"] != benchmark].copy()

## Laver en matrix
bootstrap_df = bootstrap_assets.reset_index(drop=True)

n_assets = bootstrap_df["ISIN"].nunique()
n_weeks = 4
n_scenarios_per_asset = len(bootstrap_df) // n_assets # = 1000

asset_list = bootstrap_df["ISIN"].unique()
terminal_returns = np.zeros((n_scenarios_per_asset, n_assets))

for j, isin in enumerate(asset_list):
    df_asset = bootstrap_df.loc[bootstrap_df["ISIN"] == isin, ["Week 1", "Week 2", "Week 3", "Week 4"]]
    cum_ret = (1 + df_asset).prod(axis=1) - 1
    terminal_returns[:, j] = cum_ret.values

## Formuler CVaR model

# ---------------------------
# Inputs
# ---------------------------
S, n_assets = terminal_returns.shape
P_matrix = 1.0 + terminal_returns
V0 = 1.0
alpha = 0.99
p = np.ones(S) / S

# Compute benchmark CVaR
terminal_returns_bench = (1 + bootstrap_benchmark[["Week 1","Week 2","Week 3","Week 4"]]).prod(axis=1).values - 1
terminal_returns_bench = terminal_returns_bench[:S]  # match portfolio scenario count


# Benchmark CVaR
bench_losses = V0 - (1.0 + terminal_returns_bench)
VaR_bench = np.quantile(bench_losses, alpha)
excess_bench = np.maximum(0, bench_losses - VaR_bench)
CVaR_bench = VaR_bench + np.sum(p * excess_bench) / (1 - alpha)
print("Benchmark CVaR:", CVaR_bench)

# ---------------------------
# Scenario 1: Minimize CVaR (w1)
# ---------------------------
def portfolio_cvar_obj(xvars):
    x = xvars[:n_assets]
    VaR = xvars[n_assets]
    excess = xvars[n_assets+1:]
    return VaR + np.sum(p * excess) / (1 - alpha)

def budget_constraint(xvars):
    x = xvars[:n_assets]
    return np.sum(x) - V0

def excess_constraints(xvars):
    x = xvars[:n_assets]
    VaR = xvars[n_assets]
    excess = xvars[n_assets+1:]
    losses = V0 - (P_matrix @ x)
    return excess - (losses - VaR)

def exp_ret_constraint(xvars):
    x = xvars[:n_assets]
    port_mean = (terminal_returns @ x).mean()
    bench_mean = terminal_returns_bench.mean()
    return port_mean - bench_mean

bounds = [(0, None)] * n_assets + [(None, None)] + [(0, None)] * S
x0 = np.concatenate([np.full(n_assets, V0/n_assets), [0.0], np.zeros(S)])

res1 = minimize(
    portfolio_cvar_obj,
    x0,
    method='SLSQP',
    bounds=bounds,
    constraints=[
        {'type': 'eq', 'fun': budget_constraint},
        {'type': 'ineq', 'fun': excess_constraints},
        {'type': 'ineq', 'fun': exp_ret_constraint}
    ]
)

x1_opt = res1.x[:n_assets]
VaR1_opt = res1.x[n_assets]
excess1_opt = res1.x[n_assets+1:]
CVaR1_opt = VaR1_opt + np.sum(p * excess1_opt) / (1 - alpha)
print("\nScenario 1: Minimize CVaR")
print("Success:", res1.success)
print("Optimal CVaR:", CVaR1_opt)
weights_w1 = pd.Series(x1_opt, index=asset_list)
print(weights_w1[weights_w1>1e-6].sort_values(ascending=False))

# ---------------------------
# Scenario 2: Maximize return subject to CVaR <= benchmark (w2)
# ---------------------------
def portfolio_ret_obj(xvars):
    x = xvars[:n_assets]
    port_mean = (terminal_returns @ x).mean()
    return -port_mean  # maximize -> minimize negative

def cvar_constraint(xvars):
    x = xvars[:n_assets]
    VaR = xvars[n_assets]
    excess = xvars[n_assets+1:]
    CVaR_val = VaR + np.sum(p * excess) / (1 - alpha)
    return CVaR_bench - CVaR_val  # CVaR <= benchmark

res2 = minimize(
    portfolio_ret_obj,
    x0,
    method='SLSQP',
    bounds=bounds,
    constraints=[
        {'type': 'eq', 'fun': budget_constraint},
        {'type': 'ineq', 'fun': excess_constraints},
        {'type': 'ineq', 'fun': cvar_constraint}
    ]
)

x2_opt = res2.x[:n_assets]
VaR2_opt = res2.x[n_assets]
excess2_opt = res2.x[n_assets+1:]
CVaR2_opt = VaR2_opt + np.sum(p * excess2_opt) / (1 - alpha)
print("\nScenario 2: Maximize return subject to CVaR <= benchmark")
print("Success:", res2.success)
print("Optimal expected return:", (terminal_returns @ x2_opt).mean())
print("Optimal CVaR:", CVaR2_opt)
weights_w2 = pd.Series(x2_opt, index=asset_list)
print(weights_w2[weights_w2>1e-6].sort_values(ascending=False))

##### PLOT PERFORMANCE
# Exclude benchmark from asset list
portfolio_assets = [i for i in isin_ter.keys() if i != benchmark]

perf_start = pd.to_datetime("2019-01-16")
perf_end = pd.to_datetime("2025-07-23")

returns_perf = jyske_ret.loc[(jyske_ret.index >= perf_start) & (jyske_ret.index <= perf_end), portfolio_assets]
returns_perf_bm = jyske_ret.loc[(jyske_ret.index >= perf_start) & (jyske_ret.index <= perf_end), [benchmark]]



# Portfolio returns
port1_returns = returns_perf.dot(x1_opt).fillna(0)
port2_returns = returns_perf.dot(x2_opt).fillna(0)
bench_returns = returns_perf_bm.squeeze().fillna(0)  # convert to Series

# Cumulative returns
cum_ret1 = ((1 + port1_returns).cumprod()) * 100 / ((1 + port1_returns).cumprod().iloc[0])
cum_ret2 = ((1 + port2_returns).cumprod()) * 100 / ((1 + port2_returns).cumprod().iloc[0])
cum_bench = ((1 + bench_returns).cumprod()) * 100 / ((1 + bench_returns).cumprod().iloc[0])


# Plot
plt.figure(figsize=(10,6))
plt.plot(cum_ret1.index, cum_ret1, label="Min CVaR")
plt.plot(cum_ret2.index, cum_ret2, label="Max Return w/ CVaR ≤ Benchmark")
plt.plot(cum_bench.index, cum_bench, label=f"Benchmark", linestyle='--', color='black')

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

def sharpe_ratio(ann_returns, ann_std, risk_free_rate=0.0):
    excess_returns = ann_returns - risk_free_rate
    return excess_returns.mean() / ann_std


periods_per_year = 52  

summary_stats = {}

for name, ret_series in zip(
    ['Min CVaR', 'Max Return w/ CVaR ≤ Benchmark', 'Benchmark'],
    [port1_returns, port2_returns, bench_returns]
):
    mean = ret_series.mean() * periods_per_year
    std = ret_series.std() * np.sqrt(periods_per_year)
    ann_ret = annualized_return(ret_series, periods_per_year)
    sharpe = sharpe_ratio(ann_ret, std, risk_free_rate = 0.0225)
    var95 = abs(value_at_risk(ret_series, alpha=0.05))
    cvar95 = abs(calculate_cvar(ret_series, alpha=0.05))
    
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