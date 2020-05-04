import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from data_import import getdata
from random_portfolio import random_portfolios
from efficient_frontier import efficient_frontier
from datetime import datetime


# Parameters
is_display_random = True
risk_free_rate = 0

# Stock data from web (yahoo) in pd.DataFrame
tickets = ['AAPL', 'MSFT', 'AMZN', 'GOOG']
stock_prices = getdata(tickers=tickets, start=datetime(2014, 1, 1),
                       end=datetime(2018, 1, 1), to_csv=False)
stock_prices = stock_prices.reset_index()

# Alternative method to read data: getdata(..to_csv=True) generates a csv in ./stock_data
# path='./stock_data/stock_data.csv'
# stock_prices = pd.read_csv(path)
# stock_prices = stock_prices.reset_index()

def get_simple_return(prices: pd.DataFrame) -> pd.DataFrame:
  # Dataframe with simple returns excluding the column 'Date'
  prices = prices.iloc[::1, 1:]
  returns = prices.pct_change()
  return returns

# Annualized returns and volatility
returns = get_simple_return(stock_prices)
annual_asset_mean = returns.mean() * 252
annual_asset_std = returns.std(ddof=0) * np.sqrt(252)

if is_display_random:
  # Random portfolios
  df_random = random_portfolios(returns=returns, total=5000)
  df_random['Sharpe Ratio'] = (
      df_random['Returns'] - risk_free_rate) / df_random['Volatility']
  # max sharpe random portfolio
  max_sharpe_pos = np.argmax(df_random['Sharpe Ratio'])
  max_sharpe_ret = df_random.iloc[max_sharpe_pos, 0]
  max_sharpe_vol = df_random.iloc[max_sharpe_pos, 1]
  max_sharpe_weight = df_random.iloc[max_sharpe_pos, 2:]
  max_sharpe = df_random['Sharpe Ratio'][max_sharpe_pos]
  # min vol random portfolio
  min_vol_pos = np.argmin(df_random['Volatility'])
  min_vol_ret = df_random.iloc[min_vol_pos, 0]
  min_vol_vol = df_random.iloc[min_vol_pos, 1]
  min_vol_weight = df_random.iloc[min_vol_pos, 2:]

# efficient frontier using scipy.optimize solver
ef_vol, ef_ret, ef_w, opt_max_sharpe, opt_max_sharpe_w = efficient_frontier(
    returns=returns, risk_free_rate=risk_free_rate)
opt_min_vol_pos = ef_vol.index(min(ef_vol))
opt_min_vol, opt_min_ret  = ef_vol[opt_min_vol_pos], ef_ret[opt_min_vol_pos]
opt_max_sharpe_ret = np.dot(opt_max_sharpe_w, annual_asset_mean)
opt_max_sharpe_vol = np.sqrt(
    np.dot(opt_max_sharpe_w, np.dot(opt_max_sharpe_w, returns.cov()))) * np.sqrt(252)

#####################################################
# Drawing efficient frontier with annualized returns and annualized volatility

if is_display_random:
  df_random.plot.scatter(x='Volatility', y='Returns', fontsize=12)
  plt.scatter(max_sharpe_vol, max_sharpe_ret, marker='^', color='r', s=150,
              label='Rand Port w/ Maximum Sharpe ratio: ' + str(round(max_sharpe, 4)))
  plt.scatter(min_vol_vol, min_vol_ret, marker='x', color='r',
              s=150, label='Rand Port w/ Minimum volatility: ' + str(round(min_vol_vol, 4)))

plt.plot(ef_vol, ef_ret, 'k-', markersize=2)
plt.scatter(opt_max_sharpe_vol, opt_max_sharpe_ret, marker='^', color='g',
            s=150, label='Maximum Sharpe ratio: ' + str(round(opt_max_sharpe, 4)))
plt.scatter(opt_min_vol, opt_min_ret,
            marker='x', color='g', s=150, label='Minimum volatility: ' + str(round(opt_min_vol, 4)))

plt.scatter(annual_asset_std, annual_asset_mean, marker='+', color='black', s=250)
for i, s in enumerate(returns.columns):
  plt.annotate(s, (annual_asset_std[i], annual_asset_mean[i]),
               xytext=(2, 5), textcoords='offset points')

plt.legend()
plt.ylabel('Expected Returns', fontsize=15)
plt.xlabel('Volatility', fontsize=15)
plt.title('Efficient Frontier', fontsize=30)
plt.show()
