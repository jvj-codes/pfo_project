import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import timedelta
from scipy.optimize import minimize


# training period (2013-2019 inclusive)
train_start = '2013-01-09'
train_end   = '2019-01-09'
# Load data

file_path = r"C:\Users\niels\Desktop\Kandidat\AllReturns.xlsx"
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
isins = list(isin_ter.keys())


jyske_ret_filtered = jyske_ret_filtered.loc[:, jyske_ret_filtered.columns.isin(isins)]

returns = jyske_ret_filtered.copy()  
train = returns.loc[train_start:train_end].dropna(how="all", axis=1)  
# Some calculations - perhaps necessary 
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

## STEP 5.1 -------
np.random.seed(1058)
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

## STEP 5.2 and 5.3 -------
# Fjerne benchmark

bootstrap_benchmark = bootstrap_df[bootstrap_df["ISIN"] == benchmark].copy()
bootstrap_assets = bootstrap_df[bootstrap_df["ISIN"] != benchmark].copy()

# Laver en matrix
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


# Benchmark værdier
bootstrap_benchmark = bootstrap_df_full[bootstrap_df_full["ISIN"] == benchmark].copy()

df_bm = bootstrap_benchmark.loc[:, ["Week 1", "Week 2", "Week 3", "Week 4"]]
cum_ret_bm = (1 + df_bm).prod(axis=1) - 1   # shape (L,)

# Startværdi
P0 = np.ones(n_assets)

# Sandsynligheder
p = np.ones(n_scenarios_per_asset) / n_scenarios_per_asset

# HUSK: række = scenarie, kolonne = aktiv-scenarie-afslutsværdi

# Parametre
V0 = 1.0
target_return = (1+0.02)**(4/52)
g_vector = np.full(n_scenarios_per_asset, V0 * target_return)  # terminal target = 1.02
barP = P0 * np.mean(terminal_returns, axis=0) # forventet slutpris

epsilon = 0

# BP
P0_benchmark = 1.0

benchmark_barP_float = cum_ret_bm.mean()


print("benchmark_barP:", benchmark_barP_float)

# Optimering

max_regret = 0

def regret_vector_for_x(x):
    V_term = (1 + terminal_returns) @ x
    raw_regret = (g_vector - epsilon * V0) - V_term
    return np.maximum(0, raw_regret)  # downside only

def expected_regret(x):
    return np.dot(p, regret_vector_for_x(x)) 



constraints_assets_vs_benchmark = [
    {'type': 'eq', 'fun': lambda x: np.dot(P0, x) - V0},                      # budget
    {'type': 'ineq', 'fun': lambda x: np.dot(barP, x) - benchmark_barP_float} # expected return >= benchmark
]


x0 = np.repeat(1.0 / n_assets, n_assets) 
bounds = [(0, 1)] * n_assets

res2 = minimize(expected_regret, x0, method='SLSQP', bounds=bounds, constraints=constraints_assets_vs_benchmark)
w2 = res2.x

print("Step 5.3")
print("Success:", res2.success)
print(pd.Series(w2, index=asset_list).loc[lambda s: s.abs() > 1e-6].sort_values(ascending=False))

# 5.4 ------------------------------
def expected_return(x):
    return -np.dot(barP, x)  # negative because we minimize

V_term_benchmark = 1.0 + cum_ret_bm   
benchmark_regret_vector = np.maximum(0, g_vector - V_term_benchmark)
benchmark_regret = float(np.dot(p, benchmark_regret_vector))

constraints = [
    {'type': 'eq', 'fun': lambda x: np.dot(P0, x) - V0},  # budget
    {'type': 'ineq', 'fun': lambda x: benchmark_regret - expected_regret(x)}  # expected_regret <= benchmark_regret
]

res3 = minimize(expected_return, x0, method='SLSQP', bounds=bounds, constraints=constraints,
    options={'maxiter': 5000})

w3 = res3.x
print("Step 5.4")
print("Success:", res3.success)
print("Weights (nonzero):")
print(pd.Series(w3, index=asset_list).loc[lambda s: s.abs() > 1e-6].sort_values(ascending=False))

# Feasibility test
feas_test = minimize(lambda x: 0, x0, bounds=bounds, constraints=constraints)
print("Feasibility test START ------------------ \n") 
print(feas_test.success, feas_test.message)
print("Feasibility test DONE ------------------ \n") 
# Plotting
perf_start = "2019-01-10"
perf_end = "2025-12-31"

returns_perf = jyske_ret.loc[perf_start:perf_end, jyske_ret.columns.isin(isins)]

assert benchmark in returns_perf.columns, "Benchmark data missing in performance period"

perf_assets = assets.copy()

final_asset_order = list(asset_list) + [benchmark]

w2_full = np.append(w2, 0.0)
w3_full = np.append(w3, 0.0)

returns_perf_subset = returns_perf[final_asset_order]

port2_returns = returns_perf_subset.dot(w2_full).fillna(0)
port3_returns = returns_perf_subset.dot(w3_full).fillna(0)
bench_returns = returns_perf[benchmark].fillna(0)

# Kopieret fra step 4
perf_start = '2019-01-16'
perf_end   = '2025-07-23'

portfolio_assets = [i for i in isin_ter.keys() if i != benchmark]

perf_start = pd.to_datetime("2019-01-16")
perf_end = pd.to_datetime("2025-12-31")

returns_perf = jyske_ret.loc[(jyske_ret.index >= perf_start) & (jyske_ret.index <= perf_end), portfolio_assets]
returns_perf_bm = jyske_ret.loc[(jyske_ret.index >= perf_start) & (jyske_ret.index <= perf_end), [benchmark]]

x1_opt = w2
x2_opt = w3


port1_returns = returns_perf.dot(x1_opt).fillna(0)
port2_returns = returns_perf.dot(x2_opt).fillna(0)
bench_returns = returns_perf_bm.squeeze().fillna(0)  # convert to Series

cum_ret1 =  ((1 + port1_returns).cumprod()) * 100
cum_ret2 = (1 + port2_returns).cumprod() * 100
cum_bench = (1 + bench_returns).cumprod() * 100
cum_bench[0] = 100

plt.figure(figsize=(10,6))
plt.plot(cum_ret1.index, cum_ret1, label="Step 5.3")
plt.plot(cum_ret2.index, cum_ret2, label="Step 5.4")
plt.plot(cum_bench.index, cum_bench, label=f"Benchmark", linestyle='--', color='black')

plt.title("Value Growth 2019-2025")
plt.xlabel("Date")
plt.ylabel("Portfolio Value (Indexed to 100)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

### CALCULATE PERFORMANCE OF THE STRAGIES

def annualized_return(returns, periods_per_year):
    compounded_growth = (1 + returns).prod()
    n_periods = returns.shape[0]
    return compounded_growth**(periods_per_year / n_periods) - 1

def value_at_risk(returns, alpha=0.05):
    return returns.quantile(alpha)

def calculate_cvar(returns, alpha=0.05):
    if returns.empty:
        return np.nan
    var = returns.quantile(alpha)
    return returns[returns <= var].mean()

def sharpe_ratio(ann_returns, ann_std, risk_free_rate=0.0):
    excess_returns = ann_returns - risk_free_rate
    return excess_returns.mean() / ann_std

def regret_vector_for_returns(returns):
    V_term = 1 + returns
    raw_regret = (1+0.02)**(4/52) - V_term # target = 0.02 & epsilon = 0
    return np.maximum(0, raw_regret)  # downside only

def expected_regret(returns):
    return np.mean(regret_vector_for_returns(returns))  

def agg_regret(returns):
    return np.sum(regret_vector_for_returns(returns))  

periods_per_year = 52  

summary_stats = {}

for name, ret_series in zip(
    ['Step 5.3', "Step 5.4", 'Benchmark'],
    [port1_returns, port2_returns, bench_returns]
):
    mean = ret_series.mean() * periods_per_year
    std = ret_series.std() * np.sqrt(periods_per_year)
    ann_ret = annualized_return(ret_series, periods_per_year)
    sharpe = sharpe_ratio(ann_ret, std, risk_free_rate = 0.0225)
    var95 = value_at_risk(ret_series, alpha=0.05)
    cvar95 = calculate_cvar(ret_series, alpha=0.05)
    exp_downside_regret = expected_regret(ret_series) * 100
    agg_downside_regret = agg_regret(ret_series)
    summary_stats[name] = {
        'Mean Annualized (%)': ann_ret*100,
        'STD Annualized (%)': std*100,
        'Sharpe Ratio': sharpe,
        'VaR(95%) (%)': var95*100,
        'CVaR(95%) (%)': cvar95*100,
        "Expected downside regret": exp_downside_regret
    }

pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_rows', None)
summary_df = pd.DataFrame(summary_stats).T.round(2)
print(summary_df)


print(benchmark_regret)
