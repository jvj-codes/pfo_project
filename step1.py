import numpy as np
import pandas as pd

# Settings for display
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_rows', None)

# Load data
jyske_ret = pd.read_excel('SelectedReturns.xlsx', index_col=0)
jyske_ret.index = pd.to_datetime(jyske_ret.index)

isin_to_name = {
    'DK0060259786': 'Jyske Portefølje Balanceret Akk KL',
    'DK0060259356': 'Jyske Portefølje Dæmpet Akk KL',
    'DK0060259430': 'Jyske Portefølje Stabil Akk KL',
    'DK0060259513': 'Jyske Portefølje Vækst Akk KL',
}

def annualized_stats(returns: pd.DataFrame, num_years: float, cvar_alpha=0.05) -> pd.DataFrame:
    periods_per_year = 52  # Weekly data

    # 1) Compute each 52-week “year” compounded return, then average (AAR)
    total_weeks = returns.shape[0]
    num_full_years = int(total_weeks // periods_per_year)
    annual_returns = []
    for i in range(num_full_years):
        chunk = returns.iloc[i*periods_per_year:(i+1)*periods_per_year]
        annual_ret = (1 + chunk).prod() - 1
        annual_returns.append(annual_ret)
    avg_annual_return = pd.concat(annual_returns, axis=1).mean(axis=1)

    # 2) Annualized standard deviation from weekly returns
    annualized_std = returns.std() * np.sqrt(periods_per_year)

    # 3) Annualized CVaR (expected shortfall)
    weekly_var = returns.quantile(cvar_alpha)
    weekly_cvar = returns[returns.le(weekly_var)].mean()
    annualized_cvar = weekly_cvar * np.sqrt(periods_per_year)


    return pd.DataFrame({
        'Average Annual Return (%)': avg_annual_return * 100,
        'Annualized Std (%)': annualized_std * 100,
        'Annualized CVaR (%)': annualized_cvar * 100
    })

def compute_stats_for_periods(ret_df, periods_in_years):
    latest_date = ret_df.index.max()
    all_stats = {}

    for label, years in periods_in_years.items():
        num_weeks = int(52 * years)
        start_date = ret_df.index[-num_weeks]
        sliced = ret_df.loc[start_date:latest_date]
        stats = annualized_stats(sliced, years)
        stats.index = stats.index.to_series().map(isin_to_name)
        all_stats[label] = stats.round(2)

    return all_stats

periods = {
    '3Y': 3,
    '5Y': 5,
    '12Y': 12
}

stats_dict = compute_stats_for_periods(jyske_ret, periods)

for label, df in stats_dict.items():
    print(f"\nAnnualized Statistics {label}:")
    print(df)
