import numpy as np
import pandas as pd


def transform_xy(windows, classdict):
    """
    Input:
        windows -- list of dataframes of all one class 
    Output:
        Xdata -- arrays of xyz data of each window stacked together
        ydata -- integer class labels for each window
    """
    positions = ['accX', 'accY', 'accZ']
    Xdata, ydata = [], []
    for window in windows:
        alldata = window[positions].to_numpy()
        alldata = np.append(alldata, [window[positions].mean(axis = 0)], 0)
        alldata = np.append(alldata, [window[positions].std(axis = 0)], 0)
        alldata = np.append(alldata, [window[positions].min(axis = 0)], 0)
        alldata = np.append(alldata, [window[positions].max(axis = 0)], 0)
        alldata = np.append(alldata, [window[positions].kurtosis(axis = 0)], 0)
        alldata = np.append(alldata, [window[positions].skew(axis = 0)], 0)
        Xdata.append(alldata)

        ydata.append(classdict[window['behavior'].iloc[0]])
    return np.stack(Xdata), np.asarray(ydata)