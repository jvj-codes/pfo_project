# Practical Financial Optimization – Final Project 2025

## Overview
This repository contains the full workflow and code for the **Practical Financial Optimization (PFO) final project**. The project focuses on designing, testing, and comparing investment strategies for retirement planning, drawing inspiration from a case study on advising grandparents on how to invest their savings.  

The analysis follows the eight steps outlined in the project description and includes both traditional portfolio optimization methods (Markowitz) and advanced risk-focused methods (Downside Regret and CVaR). The models are implemented in Python and rely on both historical data and bootstrapped scenarios.

Course: https://kurser.ku.dk/course/nmak15000u

---

## Project Steps

### Step 1: Jyske Bank Portfolios
- Collected data on the four Jyske Bank standard portfolios.  
- Calculated annual expense ratios, average returns, volatility, and Conditional Value at Risk (CVaR) for 3-, 5-, and 12-year periods.  
- Summarized results in tables.

### Step 2: Guaranteed Investment Option
- Modeled an investment product with fixed annual payments over 10 years.  
- Compared advantages and disadvantages relative to market-based portfolios.

### Step 3: Feature Selection
- Used Investment Funnel to reduce the investment universe to 30–50 ETFs/mutual funds.  
- Reported fund details (Name, ISIN, TER, summary statistics) (covariance matrix included in the code).  
- Justified the chosen feature selection process.

### Step 4: First Optimization Model
- Ran Markowitz optimizations:
  - Maximize expected return with portfolio volatility ≤ benchmark volatility.  
  - Minimize portfolio volatility with return ≥ benchmark return.  
- Plotted the efficient frontier with both optimized portfolios and the benchmark.  
- Backtested strategies from 2019–2025.

### Step 5: Downside Regret Optimization
- Bootstrapped 1,000 four-week scenarios from 2013–2019.  
- Implemented downside regret optimization models with constraints.  
- Backtested and compared performance to the benchmark.

### Step 6: CVaR Optimization
- Bootstrapped scenarios and implemented CVaR-based optimization.  
- Ran both minimizing CVaR (with return ≥ benchmark) and maximizing return (with CVaR ≤ benchmark).  
- Compared 2019–2025 performance against benchmark.

### Step 7: Optimizations with 4-Weekly Revisions
- Selected two favorite strategies for rolling 4-week backtesting from 2019–2025.  
- Incorporated transaction costs (0.1%) in portfolio revisions.  
- Produced:
  - Stacked graph of portfolio composition over time.  
  - Portfolio value growth (ex-post vs. ex-ante scenarios).  
  - Performance comparison across two strategies and the benchmark.  

**Note:** You can directly load the optimized backtested portfolio weights by using the pickle file:  
```python
import pickle

with open("strategy_results_cvar.pkl", "rb") as f:
    results = pickle.load(f)
```
This will provide you with precomputed, optimized results without rerunning the full simulation.

### Step 8: Conclusions and Recommendations
- Summarized findings in a written report with a supporting table.  
- Presented investment recommendations tailored to the grandparents’ needs.  

---

## Repository Structure
```
├── AllReturns.xlsx              # .xlsx file containing all returns
├── SelectedReturns.xlsx         # .xlsx file containing selected returns
├── step*.py                     # .py files for each step *=1,3,4,5,6,7.
├── strategy_results_cvar.pkl    # Precomputed optimized backtested weights (Step 7)
├── PFO_FP_QUESTIONS.pdf         # Questions for the project
├── PFO2025.pdf                  # Final project report
└── README.md                    # Project documentation
```

---

## Requirements
- Python 3.10+  
- Libraries: `numpy`, `pandas`, `matplotlib`, `scipy`, `pickle`, etc.  

---

## Usage
1. Clone the repository.  
2. Run steps in order 1-7. 
3. For Step 7, either rerun the optimization loop or load the precomputed results with `strategy_results_cvar.pkl`.  
4. Review the report PF02025.pdf for conclusions and recommendations.  
