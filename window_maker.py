import pandas as pd
import numpy as np
import random
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit


def class_divider(df, window_size):
    majoritydf = df[(df['behavior']=='s') | (df['behavior']=='l')]
    minoritydf = df[(df['behavior']!='s') & (df['behavior']!='l')]
    maj_windows , maj_classes = pull_window(majoritydf, window_size)
    min_windows , min_classes = pull_window(minoritydf, window_size)
    all_windows = maj_windows+min_windows
    all_classes = maj_classes+min_classes
    return all_windows, all_classes

#df[[ behavior,acc_x,acc_y,acc_z]].describe()

def pull_window(df, window_size, class_list):
    """
    Input: 
    df -- dataframe of cleaned input data, likely from a csv
    window_size -- number of rows of data to convert to 1 row for AcceleRater (25 = 1sec)
    Output:
    windows -- list of lists of accel data (EX:[x,y,z,...,x,y,z,class_label])
    allclasses -- list of the behavior classes that are present in the windows
    """
    classes = []
    windows = []
    number_of_rows_minus_window = df.shape[0] - window_size + 1
    if window_size > df.shape[0]:
        raise ValueError('Window larger than data given')
    for i in range(0, number_of_rows_minus_window, window_size):
        window = df[i:i+window_size]
        if len(set(window.behavior)) != 1:
            continue
        if len(set(np.ediff1d(window.input_index))) != 1:
            continue
        if window.iloc[0]['behavior'] not in class_list:
            continue
        windows.append(window)
        classes.append(window.iloc[0]['behavior'])
    allclasses = set(classes)
    print("Windows pulled")
    return windows, list(allclasses)


def rando_maker(df, window_size, class_list):
    allclasswins = []
    classcodes = []
    for classes in class_list:
        classdf, clist = pull_window(df, window_size, classes)
        rands = random.sample(classdf, 500)
        classcodes += clist
        allclasswins += rands
    return allclasswins, classcodes


# add a classifier list for classes to pay attention to 
# def pull_window(df, window_size):
#     """
#     Input: 
#     df -- dataframe of cleaned input data, likely from a csv
#     window_size -- number of rows of data to convert to 1 row for AcceleRater (25 = 1sec)
#     Output:
#     windows -- list of lists of accel data (EX:[x,y,z,...,x,y,z,class_label])
#     allclasses -- list of the behavior classes that are present in the windows
#     """
#     classes = []
#     windows = []
#     number_of_rows_minus_window = df.shape[0] - window_size + 1

#     if window_size > df.shape[0]:
#         raise ValueError('Window larger than data given')
#     if df.iloc[0]['behavior'] == 's' or df.iloc[0]['behavior'] == 'l':
#         for i in range(0, number_of_rows_minus_window, window_size):
#             window = df[i:i+window_size]
#             if len(set(window.behavior)) != 1:
#                 continue
#             if len(set(np.ediff1d(window.input_index))) != 1:
#                 continue
#             windows.append(window)
#             classes.append(window.iloc[0]['behavior'])
#             #
#     else:
#         for i in range(0, number_of_rows_minus_window, 1):
#             window = df[i:i+window_size]
#             if len(set(window.behavior)) != 1:
#                 continue
#             if len(set(np.ediff1d(window.input_index))) != 1:
#                 continue
#             windows.append(window)
#             classes.append(window.iloc[0]['behavior'])
#             #
#     allclasses = set(classes)
#     print("Windows pulled")
#     return windows, list(allclasses)


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
        ydata.append(classdict[window['behavior'].iloc[0]])
        
    return np.stack(Xdata), np.asarray(ydata)
    

def transform_xy(windows, classdict):
    """
    Input:
        windows -- list of dataframes of all one class 
    Output:
        Xdata -- arrays of xyz data of each window stacked together
        ydata -- integer class labels for each window
    """
    #positions = ['accX', 'accY', 'accZ']
    Xdata, ydata = [], []
    for window in windows:
        Xdata.append(np.array([window.accX.mean(), window.accY.mean()]))
        ydata.append(classdict[window['behavior'].iloc[0]])
        
    return np.stack(Xdata), np.asarray(ydata)




#OLD method
def reduce_dimensions(Xdata, ydata):
    nsamples, nx, ny = Xdata.shape
    Xdata2d = Xdata.reshape((nsamples,nx*ny))
    X_train, X_test, y_train, y_test = train_test_split(
        Xdata2d, ydata, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test
#



def reduce_dim_strat(Xdata,ydata):
    nsamples, nx, ny = Xdata.shape
    Xdata2d = Xdata.reshape((nsamples,nx*ny))
    stshsp = StratifiedShuffleSplit(n_splits= 1, test_size =0.2, random_state=42)
    train_index, test_index = next(stshsp.split(Xdata2d,ydata))
    x_train, x_test = Xdata2d[train_index], Xdata2d[test_index]
    y_train, y_test = ydata[train_index], ydata[test_index]

    return x_train, x_test, y_train, y_test


# def main():


# if __name__=="__main__":
#     main()

