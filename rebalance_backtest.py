import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from efficient_frontier_strat import efficient_frontier_strat


##############################
# Evaluate the weight allocations, returns of different strategies for rebalance

# ##Generation of the test_data.csv
# from data_import import getdata
# test = ['TSLA', 'MSFT', 'FB', 'TWTR']
# start = datetime(2014, 12, 31)
# end = datetime(2017, 12, 31)
# getdata(test, start, end, True)

# read prices data
path='./stock_data/test_data.csv'
#path='./stock_data/20-05-03_165042.csv'

prices = pd.read_csv(path)

# Set up
is_save_weight_allocations_to_csv = False

# Monthly portfolio rebalance considering 252 past trading days 
risk_free_rate = 0.0
window = 252
first_trading_date = datetime(2016, 1, 1)
prices.Date = pd.to_datetime(prices.Date)
num_assets = len(prices.columns[1:])

### Determine the first trading day of every month
first_tradeday_in_month = prices.groupby(pd.Grouper(key = 'Date', freq = 'M')).Date.first()
rebalance_tradeday_monthly = first_tradeday_in_month[first_tradeday_in_month > first_trading_date]
rebalance_tradeday_monthly_pos = prices[prices.Date.isin(rebalance_tradeday_monthly)].index.tolist()

### Evaluate the new portfolio allocations for first trading day every month 
### with the optimization with 
# Using the efficient_frontier_strat function to determine the allocation of assets
record_sharpe = []
record_min_vol = []
for i in rebalance_tradeday_monthly_pos:
  seleted = prices.iloc[i-window:i, :]
  record_sharpe.append(efficient_frontier_strat(seleted, 'sharpe', risk_free_rate))
  record_min_vol.append(efficient_frontier_strat(seleted, 'min_vol', risk_free_rate))
df_sharpe = pd.DataFrame(record_sharpe)
df_min_vol = pd.DataFrame(record_min_vol)

## Output to csv for record
if is_save_weight_allocations_to_csv:
  path = './weight_allocations/df_sharpe_' + datetime.now().strftime('%y-%m-%d_%H%M%S') + '.csv'
  df_sharpe.to_csv(path, index = False, header=True)
  path = './weight_allocations/df_min_vol_' + datetime.now().strftime('%y-%m-%d_%H%M%S') + '.csv'
  df_min_vol.to_csv(path, index = False, header=True)

## This function evaluate the cumulative returns, portfolio annualized returns and volatilities,
## sharpe ratio.
def get_portfolio_stat(prices: pd.DataFrame, weight: list, rebalance_tradeday_pos: list, risk_free_rate: float = 0.0) -> tuple:
  ret_plus_1 = prices.iloc[:,1:].pct_change() + 1
  # Set up the initial allocations of assets
  new_weight = weight[0] * 1
  cumulative_ret = [pd.DataFrame([new_weight], index=[prices.Date.iloc[rebalance_tradeday_pos[0]]], columns=prices.columns[1:])]
  for i in range(1, len(rebalance_tradeday_pos)):
    # Calculate the cumulative returns of assets with new weight 
    start = rebalance_tradeday_pos[i-1] + 1; end = rebalance_tradeday_pos[i] + 1
    # Calculate the current portfolio value and determine the new weights of assets
    cumulative_ret.append((ret_plus_1.iloc[start:end, :].cumprod() * new_weight))
    new_weight = weight[i] * cumulative_ret[-1].tail(1).sum(axis=1).values
  cumulative_ret.append((ret_plus_1.iloc[rebalance_tradeday_pos[-1]+1:, :].cumprod() * new_weight))
  # Merge the list of df into a single one
  cumulative_ret = pd.concat(cumulative_ret)
  cumulative_ret_history = cumulative_ret.sum(axis=1)
  cumulative_ret_history.index = prices.Date[rebalance_tradeday_pos[0]:]
  # Portfolio daily returns under this strategy
  port_ret = cumulative_ret_history.pct_change()[1:]
  annualized_port_return = cumulative_ret_history[-1] ** (252/(len(port_ret))) - 1
  annualized_port_std = np.sqrt(252) * port_ret.std(ddof=0) # ddof=0 for population std
  sharpe_ratio = annualized_port_return / annualized_port_std
  port_stat = {'cumulative_ret': cumulative_ret_history[-1] - 1, 
              'annual_ret': annualized_port_return,
              'annual_std': annualized_port_std,
              'sharpe_ratio': sharpe_ratio}
  return port_stat, cumulative_ret_history

### Get the portfolio cumulative returns with different strategies
# With the optimized 
sharpe_weight = df_sharpe.w.to_list()
minvol_weight = df_min_vol.w.to_list()
max_sharpe_port_stat, max_sharpe_cumulative_ret_history = get_portfolio_stat(
    prices, sharpe_weight, rebalance_tradeday_monthly_pos, risk_free_rate)
min_vol_port_stat, min_vol_cumulative_ret_history = get_portfolio_stat(
    prices, minvol_weight, rebalance_tradeday_monthly_pos, risk_free_rate)

# Naive diversification w/o rebalance (1/N Portfolio) 
weight = [[1/num_assets for _ in range(num_assets)]]
naive_port_stat, naive_port_cumulative_ret_history = get_portfolio_stat(
    prices, weight, [rebalance_tradeday_monthly_pos[0]], risk_free_rate)

# Naive diversification w/ rebalance (1/N Portfolio)
# Rebalance to 1/N portfolio monthly according to adj. close price on last trading day on last month
# Evaluate the weight allocations every month
naive_rebalance_weight = [[1/num_assets for _ in range(num_assets)]  for _ in range(len(rebalance_tradeday_monthly_pos))]
naive_port_rebalance_stat, naive_port_rebalance_cumulative_ret_history = get_portfolio_stat(
    prices, naive_rebalance_weight, rebalance_tradeday_monthly_pos, risk_free_rate)

# Print the porfolio statistic of different strategies
print('Max sharpe ratio portfolio stats:\n', max_sharpe_port_stat)
print('Min volatility portfolio stats:\n', min_vol_port_stat)
print('Naive diversification w/o rebalance (1/N Portfolio):\n', naive_port_stat)
print('Naive diversification w/ rebalance (1/N Portfolio):\n', naive_port_rebalance_stat)

###############################################################
# Drawing

# Firgure 1. Cumulative returns with different strategies
plt.figure()
plt.plot(max_sharpe_cumulative_ret_history, label='Max sharpe ratio')
plt.plot(min_vol_cumulative_ret_history, label='Min volatility')
plt.plot(naive_port_cumulative_ret_history, label='Naive diversification w/o rebalance (1/N Portfolio)')
plt.plot(naive_port_rebalance_cumulative_ret_history, label='Naive diversification w rebalance (1/N Portfolio)')
plt.legend()
plt.ylabel('Cumulative returns', fontsize=15)
plt.title('Cumulative returns with different strategies', fontsize=15)

# Figure 2. Assets allocations (Max Sharpe Ratio)
plt.figure()
weights = np.array(df_sharpe.w.to_list()).T
ind = np.arange(len(weights[0]))
lst_bar = [plt.bar(ind, weights[0])]
for i in range(1, num_assets):
  lst_bar.append(plt.bar(ind, weights[i], bottom=sum(weights[:i])))
plt.ylabel('Allocation', fontsize=15)
plt.title('Assets allocations (Max Sharpe Ratio)', fontsize=15)
plt.xticks(ind, rebalance_tradeday_monthly.dt.strftime('%y/%m/%d').to_list(), rotation=90)
plt.yticks(np.arange(0, 1.1, 0.1))
plt.legend((b[0] for b in lst_bar), (prices.columns[1:].to_list()))

# Figure 3. Assets allocations (Min Volatility)
plt.figure()
weights = np.array(df_min_vol.w.to_list()).T
ind = np.arange(len(weights[0]))
lst_bar = [plt.bar(ind, weights[0])]
for i in range(1, num_assets):
  lst_bar.append(plt.bar(ind, weights[i], bottom=sum(weights[:i])))
plt.ylabel('Allocation', fontsize=15)
plt.title('Assets allocations (Min Volatility)', fontsize=15)
plt.xticks(ind, rebalance_tradeday_monthly.dt.strftime('%y/%m/%d').to_list(), rotation=90)
plt.yticks(np.arange(0, 1.1, 0.1))
plt.legend((b[0] for b in lst_bar), (prices.columns[1:].to_list()))
plt.show()

