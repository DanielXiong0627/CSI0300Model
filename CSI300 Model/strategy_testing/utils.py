
import numpy as np


# utilities for normalization
def max_range_norm(column, time_diff):
    ''' column: data we want to normalize. time_diff: if time_diff is 5, then 
    we calculate the current/(max-min of last 30 days) based on the past 5 days. Time diff should always be positive.
    returns: array with the same dimensions with modified entries.'''
    temp = []
    if time_diff>0:
        for i in range(len(column)):
            if column.iloc[i] is None:
                temp.append(None)
            else:
                if i<time_diff:
                    temp.append(None)
                else:
                    temp.append(column.iloc[i]/
                                (np.max(column.iloc[i-time_diff:i+1])-np.min(column.iloc[i-time_diff:i+1])))
    return temp

def minimax(column, time_diff):
    '''minimax normalization. column: data we want to normalize. time_diff: if time_diff is 5, then 
    we calculate the minimax based on the past 5 days. Time diff should always be positive.
    returns: array with the same dimensions with modified entries.'''
    temp = []
    if time_diff>0:
        for i in range(len(column)):
            if column.iloc[i] is None:
                temp.append(None)
            else:
                if i<time_diff:
                    temp.append(None)
                else:
                    temp.append((column.iloc[i]-np.min(column.iloc[i-time_diff:i+1]))/
                                (np.max(column.iloc[i-time_diff:i+1])-np.min(column.iloc[i-time_diff:i+1])))

    return temp

def get_percentile(data, percentile, time):
    '''calculates the percentile within the given time period for an entire column'''
    temp = []
    data=np.array(data).astype(float)
    for i in np.arange(len(data)):
        if i-time<0 or np.isnan(data[i-time:i+1]).any():
            temp.append(None)
        else:
            temp.append(sorted(data[i-time:i+1])[int(percentile*time)])
    return temp

def z_score(data, mean, stddev):
    '''calculates z_score'''
    temp = []
    for i in np.arange(len(data)):
        if data[i] is None or mean[i] is None or stddev[i] is None:
            temp.append(None)
        else:
            temp.append((data[i]-mean[i])/stddev[i])
    return temp
