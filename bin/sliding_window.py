import pandas as pd
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import StratifiedShuffleSplit



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



def leaping_window(df, window_size):
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
    for i in range(0, number_of_rows_minus_window, window_size):
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
        ydata -- binary multiclass labels of each class present in each window
    """
    positions = ['accX', 'accY', 'accZ']
    Xdata, ydata = [], []
    for window in windows:
        Xdata.append(window[positions].to_numpy())
        bclass = list(window.behavior.unique().sum())
        numlist = [classdict[yval] for yval in bclass]
        ydata.append(numlist)
    return np.stack(Xdata), MultiLabelBinarizer().fit_transform(ydata)


def reduce_dim_strat(Xdata,ydata):
    nsamples, nx, ny = Xdata.shape
    Xdata2d = Xdata.reshape((nsamples,nx*ny))
    stshsp = StratifiedShuffleSplit(n_splits= 1, test_size =0.2, random_state=42)
    train_index, test_index = next(stshsp.split(Xdata2d,ydata))
    x_train, x_test = Xdata2d[train_index], Xdata2d[test_index]
    y_train, y_test = ydata[train_index], ydata[test_index]

    return x_train, x_test, y_train, y_test