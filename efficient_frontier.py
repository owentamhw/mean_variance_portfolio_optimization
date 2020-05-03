import pandas as pd
import numpy as np
import scipy.optimize as solver
from functools import reduce

# optimized based on annualized returns and volatilities
def efficient_frontier(returns: pd.DataFrame, risk_free_rate: float):
  efficient_frontier_ret = []
  efficient_frontier_vol = []
  efficient_frontier_weight = []
  
  min_vol = float("inf")
  cov = returns.cov()               #daily
  expected_ret = returns.mean()     #daily
  num_assets = len(returns.columns)

  # optimizing vol with given expected returns
  def vol(w):
    return np.sqrt(reduce(np.dot, [w.T, cov, w])) * np.sqrt(252)

  #setup for scipy.optimize
  ini_guess = np.array(num_assets*[1/num_assets])
  bounds = tuple((0.0, 1.0) for x in range(num_assets))
  target_return = np.linspace(
      max(expected_ret*252), min(expected_ret*252), 1000)

  for target in target_return:
    # sum of w = 1
    # sum of w*r = target return
    constraints = [{'type': 'eq', 'fun': lambda x: sum(x) - 1}, 
        {'type': 'eq', 'fun': lambda x: sum(x * expected_ret)*252 - target}]
    outcome = solver.minimize(
        fun=vol, x0=ini_guess, constraints=constraints, bounds=bounds, method='SLSQP')
    if outcome.fun < min_vol: min_vol = outcome.fun
    else: break
    efficient_frontier_ret.append(target)
    efficient_frontier_vol.append(outcome.fun)
    efficient_frontier_weight.append(outcome.x)

  # optimizing for max sharpe ratio
  def negative_sharpe(w, expected_ret, cov, risk_free_rate):
    vol = np.sqrt(reduce(np.dot, [w.T, cov, w])) * np.sqrt(252)
    ret = np.dot(w, expected_ret) * 252
    return -(ret - risk_free_rate) / vol

  args = (expected_ret, cov, risk_free_rate)
  constraints = [{'type': 'eq', 'fun': lambda x: sum(x) - 1}]
  min_neg_sharpe = solver.minimize(fun=negative_sharpe, x0=ini_guess,
                                   args=args, constraints=constraints, bounds=bounds, method='SLSQP')
  opt_max_sharpe, opt_max_sharpe_weights = -min_neg_sharpe.fun, min_neg_sharpe.x

  return efficient_frontier_vol, efficient_frontier_ret, efficient_frontier_weight, opt_max_sharpe, opt_max_sharpe_weights