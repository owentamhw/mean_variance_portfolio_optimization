# Mean variance portfolio optimization
This project aims to optimize the portfolio returns using Markowitz’s portfolio optimization and the Modern Portfolio Theory.

## Folders
### [results](results)
Stores figures plotted by the project.
- Cumulative returns
- Weight allocation of assets
- Drawdown
### [stock_data](stock_data)
Stores the daily prices used for portfolio optimization which are collected through [**data_import.py**](data_import.py).
### [weight_allocation](weight_allocation)
Stores the evaluated weight allocation under different strategies.

## Python code
### [**data_import.py**](data_import.py)
To obtain stock prices data from Yahoo using pandas_datareader with the option to output as .csv file.
### [**display_efficient_frontier_test.py**](display_efficient_frontier_test.py)
Plot the efficient frontier with given prices data.
Both Monte Carlo simulation ([**random_portfolio.py**](random_portfolio.py)) and minimize solver ([**efficient_frontier.py**](efficient_frontier.py)) were used to evaluate the portfolios with highest Sharpe ratio or minimum volatility.
### [**efficient_frontier_strat.py**](efficient_frontier_strat.py)
Functions of implementation of strategies (Maximum sharpe ratio, minimum volatility). Returns dictionary with optimized portfolio weight allocation.
### [**efficient_frontier.py**](efficient_frontier.py)
The main body using scipy.optimize.minimize to solve for optimized weight allocation and evaluation of efficient frontier.
### [**portfolio_statistic.py**](portfolio_statistic.py)
This includes functions for the portfolio statistics evaluations.
### [**random_portfolio.py**](random_portfolio.py)
Random generations of portfolios with different random asset allocation (Monte Carlo).
### [**rebalance_backtest.py**](rebalance_backtest.py)
Implementation of tradings strategies following the mean variance portfolio optimization. 
- Real historical stock prices were obtained using [**data_import.py**](data_import.py)
- [**efficient_frontier_strat.py**](efficient_frontier_strat.py) are used to obtain optimized weight allocation at the first trading day of each month based on the prices of past 252 trading days.
- get_portfolio_stat simulates the monthly rebalance of portfolio throughout the whole period of time.
- Naive diversification strategies (with and without rebalance) were simulated for performance comparison.
- Figures on portfolio cumulative returns, drawdowns and weight allocation of assets were plotted.
