import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from efficient_frontier import efficient_frontier


def get_simple_return(prices: pd.DataFrame) -> pd.DataFrame:
  # Dataframe with simple returns excluding the column 'Date'
  prices = prices.iloc[::1, 1:]
  returns = prices.pct_change()
  return returns

# Strategy: Max Sharpe ratio
def ef_strat_max_sharpe(returns: pd.DataFrame, risk_free_rate: float=0.0) -> dict:
  _, _, _, opt_max_sharpe, opt_max_sharpe_w = efficient_frontier(
    returns=returns, risk_free_rate=risk_free_rate)
  
  w = opt_max_sharpe_w
  sharpe = opt_max_sharpe
  ret = np.dot(w, returns.mean()) * 252
  vol = np.sqrt(np.dot(w, np.dot(w, returns.cov()))) * np.sqrt(252)
  
  return {'ret': ret, 'vol': vol, 'w': w, 'sharpe': sharpe}

# Strategy: Minimum volatility
def ef_strat_min_vol(returns: pd.DataFrame, risk_free_rate: float=0.0) -> dict:
  ef_vol, ef_ret, ef_w, _, _ = efficient_frontier(
    returns=returns, risk_free_rate=risk_free_rate)
  
  pos = ef_vol.index(min(ef_vol))
  ret, vol = ef_ret[pos], ef_vol[pos]
  w = ef_w[pos]
  sharpe = (ret - risk_free_rate) / vol
  
  return {'ret': ret, 'vol': vol, 'w': w, 'sharpe': sharpe}

def efficient_frontier_strat(prices: pd.DataFrame, strategy: str = 'sharpe', risk_free_rate: float=0.0) -> dict:
  # returns
  returns = get_simple_return(prices)

  # strategy 
  if strategy == 'sharpe':
    return ef_strat_max_sharpe(returns, risk_free_rate)
  
  if strategy == 'min_vol':
    return ef_strat_min_vol(returns, risk_free_rate)

  return -1