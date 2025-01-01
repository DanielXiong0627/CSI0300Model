import pandas as pd
import talib as ta
import numpy as np
FILE_PATH = 'HS300data.csv'
# Preprocessing; cleans the data and calculates the inputs and labels.

# sort data based on time
data = pd.read_csv(FILE_PATH)
data = data.sort_values(by='TRADE_DATE')
data = data.set_index('TRADE_DATE')
data.index = pd.to_datetime(data.index.values, format="%Y%m%d")
print(data)
high = data['HIGH']
low = data['LOW']
close = data['CLOSE']
open = data['OPEN']
vol = data['VOL']

# calculate indicators
ema_close= ta.EMA(close, 25)
rsi = ta.RSI(close, 14)
kama = ta.KAMA(close, 15)
aroon = ta.AROONOSC(high, low, 15)
cmo = ta.CMO(close, 15)
dx = ta.DX(high,low, close,13)
cci = ta.CCI(high, low, close, 20)
chaikin = ta.AD(high, low, close, vol)
atr = ta.ATR(high, low, close, 15)
obv = ta.OBV(close,vol)
ht = ta.HT_TRENDLINE(close)
ht_period = ta.HT_DCPERIOD(close)
ht_trend = ta.HT_TRENDMODE(close)
bop = ta.BOP(open, high, low, close)
macd,a,b = ta.MACD(close)

target = open
# helper functions for normalization
def percentChange(column, time_diff):
    '''computes the percentage change. 
    column: the data column that it is computed for.
    time_diff: if time difference is 1, then we calculate tomorrow/today-1. If it is -1, then today/yesterday-1.
    returns: array with the same dimensions with modified entries.
    '''
    temp = []
    if time_diff>0:
        for i in range(len(column)):
            if i>=len(column)-time_diff:
                temp.append(None)
            else:
                temp.append(column.iloc[i+time_diff]/column.iloc[i]-1)
    if time_diff<0:
        for i in range(len(column)):
            if i<=-time_diff:
                    temp.append(None)
            else:
                temp.append(column.iloc[i]/column.iloc[i+time_diff]-1)
    return temp

def minimax(column, time_diff):
    '''minimax normalization. 
    column: data we want to normalize. 1 pd dataframe column
    time_diff: if time_diff is 5, then 
    we calculate the minimax based on the past 5 days. Time diff should always be positive.
    returns: array with the same dimensions with modified entries.'''
    temp = []
    if time_diff>0:
        for i in range(len(column)):
            if i<time_diff:
                temp.append(None)
            else:
                temp.append((column.iloc[i]-np.min(column.iloc[i-time_diff:i+1]))/
                            (np.max(column.iloc[i-time_diff:i+1])-np.min(column.iloc[i-time_diff:i+1])))

    return temp

# normalization
label = percentChange(target,1) # Percentage change tomorrow
label.insert(len(label),None)
label = label[1:]
aroon =aroon/100
print(aroon)
rsi = rsi/100
cmo = cmo/100
ht=ht/close -1
dx = dx/100
ht_period = ht_period/100
ema_close = ema_close/close-1
kama = kama/close-1
atr = percentChange(atr,-1)
chaikin = percentChange(chaikin,-1)
obv = percentChange(obv,-1)
perc_close = percentChange(close,-1)
perc_high = percentChange(high,-1)
perc_low = percentChange(low,-1)
macd = minimax(macd, 20)
perc_open = percentChange(open, -1)

# saving to csv
data = pd.DataFrame({'ht_trend':ht_trend,'ht_period':ht_period,'dx':dx, 'ema':ema_close, 'kama':kama, 'cmo':cmo, 'rsi':rsi, 
                    'cci':cci, 'chaikin':chaikin, 'atr':atr, 'obv':obv,'aroon':aroon,
                    'ht':ht, 'CLOSE':perc_close,'HIGH':perc_high, 'LOW':perc_low, 'bop':bop, 'macd':macd, 'OPEN':perc_open,
                    'date': data.index.values, 'label':label,'price':open,'close_price':close,
                    'low_price':low,'high_price':high})

data.reset_index(drop=True, inplace=True)
print(data)
data.to_csv('processedHS300.csv')
