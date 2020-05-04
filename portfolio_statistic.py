import numpy as np
import pandas as pd


def get_port_cumulative_return(prices: pd.DataFrame, weight: list, rebalance_tradeday_pos: list) -> pd.Series:
  # Set up the initial allocations of assets
  ret_plus_1 = prices.iloc[:,1:].pct_change() + 1
  new_weight = weight[0] * 1
  # cumulative_ret stores the cumulative returns of individual assets
  cumulative_ret = [pd.DataFrame([new_weight], index=[prices.Date.iloc[rebalance_tradeday_pos[0]]], columns=prices.columns[1:])]
  for i in range(1, len(rebalance_tradeday_pos)):
    # Calculate the cumulative returns of assets with new weight 
    start = rebalance_tradeday_pos[i-1] + 1; end = rebalance_tradeday_pos[i] + 1
    cumulative_ret.append((ret_plus_1.iloc[start:end, :].cumprod() * new_weight))
    # Calculate the current portfolio value and determine the new weights of assets
    new_weight = weight[i] * cumulative_ret[-1].tail(1).sum(axis=1).values
  cumulative_ret.append((ret_plus_1.iloc[rebalance_tradeday_pos[-1]+1:, :].cumprod() * new_weight))
  # Merge the list of df into a single one
  cumulative_ret = pd.concat(cumulative_ret)
  # Portfolio cumulative return history
  cumulative_ret_history = cumulative_ret.sum(axis=1)
  cumulative_ret_history.index = prices.Date[rebalance_tradeday_pos[0]:]
  return cumulative_ret_history

def get_drawdown(port_cumulative_ret_history: pd.Series) -> pd.Series:
  return port_cumulative_ret_history - port_cumulative_ret_history.cummax()
  
def get_sortino_ratio(returns: pd.Series, target: float = 0, risk_free_rate: float = 0.0) -> float:
  downside_ret_std = returns[returns < target].std()  
  return (returns.mean() - risk_free_rate)/downside_ret_std