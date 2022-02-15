import pandas as pd
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer



def slide_window(df, window_size, slide: int = 1):
    """
    Input: 
    df -- dataframe of input data, likely from a csv
    window_size -- number of rows of data to convert to 1 row for AcceleRater (25 = 1sec)
    Output:
    windows -- list of dataframes of accel data
    allclasses -- list of the behavior classes that are present in the windows
    """
  #  classes = []
    windows = []
    number_of_rows_minus_window = df.shape[0] - window_size + 1
    if window_size > df.shape[0]:
        raise ValueError('Window larger than data given')
    for i in range(0, number_of_rows_minus_window, slide):
        window = df[i:i+window_size]
        windows.append(window)
#        classes.append(list(window.Behavior.unique().sum()))
   # allclasses = set(classes)
    print("Windows pulled")
    return windows




def construct_xy(windows, classdict):
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
        Xdata.append(window[positions].to_numpy())
        bclass = list(window.Behavior.unique().sum())
        for yvals in bclass:
            numlist = [classdict[yval] for yval in yvals]
            ydata.append(numlist)
        
    return np.stack(Xdata), MultiLabelBinarizer().fit_transform(ydata)
