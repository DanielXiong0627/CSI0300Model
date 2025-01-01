import pandas as pd
import matplotlib.pyplot as plt
import talib as ta
import numpy as np
from utils import max_range_norm
from utils import minimax
from utils import get_percentile
from utils import z_score
# Code constructs the weights required for backtesting

signal = pd.read_csv('signal.csv')
signal = signal.drop(signal.columns[0],axis=1)

# approach 1: signal normalization using max,min
# norm_range = 20
# percent_period = 100
# signal = np.array(max_range_norm(signal,norm_range))
# upper = get_percentile(signal,0.8,percent_period)
# lower = get_percentile(signal,0.2,percent_period)

# weights = []
# i = 0
# for data in signal:
#     if data is None or upper[i] is None or lower[i] is None:
#         weights.append(0)
#     else:
#         # if data <upper and data>mid_up:
#         #     weights.append(0.3)
#         # elif data >lower and data<mid_down:
#         #     weights.append(-0.3)
#         if data>= upper[i]:
#             weights.append(1)
#         elif data< lower[i]:
#             weights.append(-1)
#         else:
#             weights.append(0)
#     i+=1

# approach 2: signal normalization using std and mean
time_span =20
mean_reg = ta.SMA(signal['reg'],time_span)
mean_cl = ta.SMA(signal['class'],time_span)
stddev_reg = ta.STDDEV(signal['reg'], time_span)
stddev_cl = ta.STDDEV(signal['class'], time_span)
z_reg = z_score(signal['reg'], mean_reg, stddev_reg)
z_class = z_score(signal['class'], mean_cl, stddev_cl)
weights = []
i = 0

# adding the weights
for data in signal['reg']:
    if data is None or z_reg[i] is None:
        weights.append(0)
    else:
        if z_reg[i] >=0.5 and z_reg[i] <=1:
            weights.append(0.5)
        elif z_reg[i] <=-0.5 and z_reg[i] > -1:
            weights.append(-0.5)
        elif z_reg[i]>= 1:
            weights.append(1)
        elif z_reg[i]< -1:
            weights.append(-1)
        else:
            weights.append(0)
            if z_class[i] >0.5:
                weights[i] +=0.3
            elif z_class[i] <-0.5:
                weights[i] +=-0.3
    i+=1

pd.DataFrame(weights).to_csv('weights.csv')
plt.show()