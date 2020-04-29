import pandas as pd
import numpy as np


def random_portfolios(returns: pd.DataFrame, total: int = 5000) -> pd.DataFrame:
  lst_ret = []
  lst_vol = []
  lst_weight = []
  cov = returns.cov()
  exp_ret = returns.mean()
  num_assets = len(returns.columns)

  # Evaluate corresponding returns and volatilities 
  # for each randomly generated weights
  for _ in range(total):
    w = np.random.random(num_assets)
    w /= np.sum(w)
    lst_weight.append(w)
    ret = np.dot(w, exp_ret)
    lst_ret.append(ret)
    vol = np.sqrt(np.dot(w, np.dot(cov, w)))
    lst_vol.append(vol)

  # Dataframe for storing all results
  portfolio = {'Returns': lst_ret, 'Volatility': lst_vol}
  for index, asset in enumerate(returns.columns):
    portfolio[asset + ' Weight'] = [w[index] for w in lst_weight]

  # Rearranging the columns of dataframe
  df = pd.DataFrame(portfolio)
  order = ['Returns', 'Volatility'] + \
      [asset + ' Weight' for asset in returns.columns]
  df = df[order]

  return df