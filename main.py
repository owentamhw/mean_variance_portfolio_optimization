import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from data_import import getdata
from random_portfolio import random_portfolios
from efficient_frontier import efficient_frontier
from datetime import datetime


# Stock data from web (yahoo) in pd.DataFrame
tickets = ['AAPL', 'MSFT', 'AMZN', 'GOOG']
stock_prices = getdata(tickers=tickets, start=datetime(2014, 1, 1),
    end=datetime(2018, 1, 1), to_csv=False)
stock_prices = stock_prices.reset_index()

# Alternative method to read data: getdata(..to_csv=True) generates a csv in ./stock_data
# path='./stock_data/stock_data.csv'
# stock_prices = pd.read_csv(path)

# Dataframe with simple returns excluding the column 'Date'
stock_prices_daily = stock_prices.iloc[::1, 1:]
returns = stock_prices_daily.pct_change()

# Parameters
asset_mean = returns.mean()
asset_std = returns.std()
risk_free_rate = 0

# Random portfolios
df_random = random_portfolios(returns=returns, total=2000)
df_random['Sharpe Ratio'] = (
    df_random['Returns'] - risk_free_rate) / df_random['Volatility']
# max sharpe random portfolio
max_sharpe_pos = np.argmax(df_random['Sharpe Ratio'])
max_sharpe_ret = df_random.iloc[max_sharpe_pos, 0]
max_sharpe_vol = df_random.iloc[max_sharpe_pos, 1]
max_sharpe_weight = df_random.iloc[max_sharpe_pos, 2:]
# min vol random portfolio
min_vol_pos = np.argmin(df_random['Volatility'])
min_vol_ret = df_random.iloc[min_vol_pos, 0]
min_vol_vol = df_random.iloc[min_vol_pos, 1]
min_vol_weight = df_random.iloc[min_vol_pos, 2:]

# efficient frontier using scipy.optimize solver
ef_vol, ef_ret, ef_w, opt_max_sharpe, opt_max_sharpe_w = efficient_frontier(
    returns=returns, risk_free_rate=risk_free_rate)
opt_min_vol_pos = ef_vol.index(min(ef_vol))
opt_max_sharpe_ret = np.dot(opt_max_sharpe_w, asset_mean)
opt_max_sharpe_vol = np.sqrt(
    np.dot(opt_max_sharpe_w, np.dot(opt_max_sharpe_w, returns.cov())))

#####################################################
# Drawing
df_random.plot.scatter(x='Volatility', y='Returns', fontsize=12)

plt.scatter(max_sharpe_vol, max_sharpe_ret, marker='^', color='r', s=150,
            label='Rand Port w/ Maximum Sharpe ratio: ' + str(round(df_random['Sharpe Ratio'][max_sharpe_pos], 4)))
plt.scatter(min_vol_vol, min_vol_ret, marker='x', color='r',
            s=150, label='Rand Port w/ Minimum volatility')

plt.plot(ef_vol, ef_ret, 'k-', markersize=2)
plt.scatter(opt_max_sharpe_vol, opt_max_sharpe_ret, marker='^', color='g',
            s=150, label='Maximum Sharpe ratio: ' + str(round(opt_max_sharpe, 4)))
plt.scatter(ef_vol[opt_min_vol_pos], ef_ret[opt_min_vol_pos],
            marker='x', color='g', s=150, label='Minimum volatility')

plt.scatter(asset_std, asset_mean, marker='+', color='black', s=250)
for i, s in enumerate(returns.columns):
  plt.annotate(s, (asset_std[i], asset_mean[i]),
               xytext=(2, 5), textcoords='offset points')

plt.legend()
plt.ylabel('Expected Returns', fontsize=15)
plt.xlabel('Volatility', fontsize=15)
plt.title('Efficient Frontier', fontsize=30)
plt.show()