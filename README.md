# Efficient Frontier Portfolio Optimizer

StudentID: 110612117  
Name: Chung-Yu Chang (張仲瑜), 何昕叡, 徐睿廷, 楊竣傑, 陳炤宇

## Introduction
This project builds an efficient frontier for a basket of equities using historical prices, then finds the minimum-risk portfolio and visualizes the frontier with Plotly. Core logic lives in [efficirnt_frontier_v2.py](efficirnt_frontier_v2.py), combining Monte Carlo sampling for intuition with constrained optimization for the true frontier.

Credit to https://github.com/tejtw/TEJAPI_Python_Medium_Quant/blob/main/TEJAPI_Medium%E9%87%8F%E5%8C%96%E5%88%86%E6%9E%907.ipynb
Some part of the code are from TEJAPI_Python_Medium_Quant.

## What I implemented
- Data pipeline that pulls 10 years of daily prices for seven tickers (AAPL, ASML, GOOG, META, MSFT, NVDA, TSLA) via `ffn` and converts them to returns/covariance.
- Portfolio statistics helpers: return computation, covariance extraction, portfolio standard deviation objective, and Monte Carlo sampler to sketch the frontier shape.
- Constrained optimizer (SLSQP) to find the minimum-volatility portfolio for each target return and to get the global minimum-volatility point.
- Plotly visualization combining random samples, the efficient frontier curve, and the optimal portfolio marker with hoverable weight breakdowns.

## How to run
- Environment: Python 3.x with `ffn`, `numpy`, `plotly`, `scipy`.
- Install deps (PowerShell example): `pip install ffn numpy plotly scipy`
- Execute from this folder: `python efficirnt_frontier_v2.py`
- Output: prints optimal weights/std-dev/returns to console and opens an interactive Plotly figure showing random samples, the efficient frontier, and the optimal point.

## Notes
- `maxiter` (default 10000) controls Monte Carlo sample count and SLSQP iteration cap; lower it if runs are slow.
- Tickers/start/end dates are defined near the bottom of [efficirnt_frontier_v2.py](efficirnt_frontier_v2.py); edit them to change the universe or horizon.
- Bounds keep weights in [0,1] with a full-investment constraint (sum to 1); there is no shorting or leverage in the current setup.
- Plotly opens in a browser window; if running headless, switch to `write_html` instead of `fig.show()`.
- See [2023_Linear_Algebra_Team4_Final_Project_Report.pdf](2023_Linear_Algebra_Team4_Final_Project_Report.pdf) for a detailed explanation of the methodology and results.
- See [demo.mp4](demo.mp4) for the visualization of the efficient frontier.
