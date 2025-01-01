import backtest as bt
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from bt import Backtest
from bt import run
from bt import algos
from bt import Strategy

# Backtesting. Run strategy.py first
# getting the index names to be dates. 
dates = pd.read_csv('dates.csv')
dates = dates.drop(dates.columns[0],axis=1)
signal = pd.read_csv('signal.csv')
signal = signal.drop(signal.columns[0],axis=1)

weights = pd.read_csv('weights.csv')
mapping = dict(zip(np.arange(len(weights)),dates[dates.columns[0]]))
weights = weights.drop(weights.columns[0],axis=1)
weights.rename(index=mapping, inplace=True)
weights.index = pd.to_datetime(weights.index.values)

data = pd.read_csv('labels.csv')
data =data.drop(data.columns[0],axis=1)
data.rename(index=mapping, inplace=True)
data.index = pd.to_datetime(data.index.values)
data.rename(columns={'price':'0'}, inplace=True)
print(data)

s = Strategy('s1',[bt.algos.RunDaily(),
                   bt.algos.WeighTarget(weights), bt.algos.Rebalance()])
t= bt.Backtest(s, data, integer_positions=False, initial_capital=1000000)
res = run(t)
res.plot()
res.display()
plt.show()
