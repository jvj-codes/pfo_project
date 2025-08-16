# -*- coding: utf-8 -*-
"""
Created on Sat Aug  9 14:40:07 2025

@author: nuffz
"""

import pandas as pd
import numpy as np

# ---- Parameters ----
start_date = '2013-01-09'
end_date   = '2019-01-09'
confidence_level = 0.95
alpha = 1 - confidence_level

# Load data

file_path = r"AllReturns.xlsx"
returns = pd.read_excel(file_path, index_col=0)

# Ensure index is datetime
returns.index = pd.to_datetime(returns.index)

# Filter date range 
returns_filtered = returns.loc[start_date:end_date]


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
            , 'DK0060158160': 1.44, 'DK0061553245': 0.21, 'DK0061150984': 0.22, 'DK0060227239': 0.35, 'DK0060118610': 0.57}


returns_filtered = returns_filtered.loc[:, returns_filtered.columns.isin(isin_ter.keys())]

# function: CVaR, Sharpe, etc
def calculate_cvar(returns, alpha):
    if returns.empty:
        return np.nan
    var = returns.quantile(alpha)
    return returns[returns <= var].mean()

def sharpe_ratio(returns, risk_free_rate=0.0225):
    excess_returns = returns - risk_free_rate
    return excess_returns.mean() / returns.std()

def max_drawdown(returns):
    cumulative = (1 + returns).cumprod()
    peak = cumulative.cummax()
    drawdown = (cumulative - peak) / peak
    return drawdown.min() 

def time_under_water(returns):
    cumulative = (1 + returns).cumprod()
    peak = cumulative.cummax()
    underwater = cumulative < peak
    # Count consecutive periods underwater
    tuw = []
    count = 0
    for flag in underwater:
        if flag:
            count += 1
        else:
            if count > 0:
                tuw.append(count)
                count = 0
    if count > 0:
        tuw.append(count)
    return max(tuw) if tuw else 0  

def ann_ret_net_cost(ann_ret: float, fee: float) -> float:
    return ann_ret - fee

# Dates
start = pd.to_datetime(start_date)
end   = pd.to_datetime(end_date)
first_full_year = start.year + (1 if start.date() > pd.Timestamp(start.year, 1, 1).date() else 0)
last_full_year  = end.year   - (1 if end.date() < pd.Timestamp(end.year, 12, 31).date() else 0)

# Results
results = []
annual_returns_dict = {}  

for asset in returns_filtered.columns:
    s = returns_filtered[asset]
    fee = isin_ter[asset]

    # calendar-year compounded returns 
    ann = (1 + s).resample('YE').prod() - 1

    # keep only the full years inside the interval
    ann = ann[(ann.index.year >= first_full_year) & (ann.index.year <= last_full_year)]
    
    

    # overall annualized return across the whole period
    num_years = (end - start).days / 365.25
    overall_annualized_return = (1 + s).prod() ** (1 / num_years) - 1

    # std dev of the annual returns
    annual_sd = s.std() * np.sqrt(52)

    # CVaR on annual returns
    cvar_annual = calculate_cvar(ann, alpha)

    # Sharpe on annual returns
    sharpe_annual = sharpe_ratio(ann)

    # TuW on annual returns
    TuW = time_under_water(ann)

    # max drawdown on annual returns
    mad = max_drawdown(ann)

    # Skewness
    skew = ann.skew()

    # Kurtosis
    kurt = ann.kurtosis()

    results.append({
        "ISIN": asset,
        "TER": fee,
        "Ann. Ret(%)": overall_annualized_return * 100,
        "Net Ann. Ret(%)": overall_annualized_return * 100 - fee,
        "Ann. STD(%)": annual_sd * 100,
        f"CVaR {int(confidence_level*100)}%(%)": cvar_annual * 100,
        "TuW": TuW,
        "Sharpe": sharpe_annual,
        "Maximum Drawdown(%)": mad * 100,
        "Skew": skew,
        "Ex. Kurtosis": kurt#,
        #"Num Full Years": ann.shape[0]
    })

    # store annual returns
    if not ann.empty:
        ann.index = ann.index.year
        annual_returns_dict[asset] = ann

# build DataFrames
summary_df = pd.DataFrame(results).round(2)
annual_df = pd.DataFrame(annual_returns_dict).sort_index()
# Variance Covariance
var_cov_matrix = annual_df.cov()
# Save to Excel with two sheets
out_path = "selected_asset_risk_return_summary.xlsx"
with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
    summary_df.to_excel(writer, sheet_name="Summary", index=False)
    annual_df.to_excel(writer, sheet_name="AnnualReturns",)
    var_cov_matrix.to_excel(writer, sheet_name="VarCov")

print("Saved:", out_path)
