import pandas_datareader.data as web
from datetime import datetime
import matplotlib.pyplot as plt
import pandas as pd
from typing import List

def getdata(tickers:List[str], start:datetime, end:datetime, to_csv=False):
  prices = []
  for ticker in tickers:
      adj_close = web.DataReader(name=ticker, start=start, end=end, data_source='yahoo')[['Adj Close']]
      prices.append(adj_close.assign(Ticker=ticker)[['Ticker', 'Adj Close']])
  df = pd.concat(prices)
  df = df.reset_index().pivot(index='Date', columns='Ticker', values='Adj Close')
  if to_csv:
    path = './stock_data/' + datetime.now().strftime('%y-%m-%d_%H%M%S') + '.csv'
    df.to_csv(path, index = True, header=True)
    return df
  else:
    return df
