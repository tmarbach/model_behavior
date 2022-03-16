import pandas as pd
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import RandomOverSampler, SMOTE, ADASYN



def singleclass_leaping_window(df, window_size, coi):
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
        # if window.iloc[0]['behavior'] not in class_list:
        #     continue
        windows.append(window)
        classes.append(window.iloc[0]['behavior'])
    allclasses = set(classes)
    diff = list(set(coi)-allclasses)
    if len(diff) > 0:
        print("Classes " + str(diff) + " not found in any window.")
    print("Windows pulled")
    return windows#, list(allclasses)



def slide_window(df, window_size, coi, slide: int = 1):
    """
    Input: 
    df -- dataframe of input data, likely from a csv
    window_size -- number of rows of data to convert to 1 row for AcceleRater (25 = 1sec)
    Output:
    windows -- list of dataframes of accel data
    allclasses -- list of the behavior classes that are present in the windows
    """
    classes = []
    windows = []
    number_of_rows_minus_window = df.shape[0] - window_size + 1
    if window_size > df.shape[0]:
        raise ValueError('Window larger than data given')
    for i in range(0, number_of_rows_minus_window, slide):
        window = df[i:i+window_size]
        windows.append(window)
        classes.append(list(window.Behavior.unique().sum()))
    allclasses = set(classes)
    diff = list(set(coi)-allclasses)
    if len(diff) > 0:
        print("Classes " + str(diff) + " not found in any window.")
    print("Windows pulled")
    return windows



def multiclass_leaping_window(df, window_size, coi):
    """
    Input: 
    df -- dataframe of input data, likely from a csv
    window_size -- number of rows of data to convert to 1 row for AcceleRater (25 = 1sec)
    Output:
    windows -- list of dataframes of accel data
    """
    classes = []
    windows = []
    number_of_rows_minus_window = df.shape[0] - window_size + 1
    if window_size > df.shape[0]:
        raise ValueError('Window larger than data given')
    for i in range(0, number_of_rows_minus_window, window_size):
        window = df[i:i+window_size]
        windows.append(window)
        classes.append(list(window.Behavior.unique().sum()))
    allclasses = set(classes)
    diff = list(set(coi)-allclasses)
    if len(diff) > 0:
        print("Classes " + str(diff) + " not found in any window.")
    print("Windows pulled")
    return windows




#multiclass labeling each windows, used with either sliding/leaping windows
def multilabel_xy(windows, classdict):
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


#single class labeling each window, use with pull windows
def singlelabel_xy(windows, classdict):
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


#stratified random sample of training data
def reduce_dim_strat(Xdata,ydata):
    nsamples, nx, ny = Xdata.shape
    Xdata2d = Xdata.reshape((nsamples,nx*ny))
    stshsp = StratifiedShuffleSplit(n_splits= 1, test_size =0.2, random_state=42)
    train_index, test_index = next(stshsp.split(Xdata2d,ydata))
    x_train, x_test = Xdata2d[train_index], Xdata2d[test_index]
    y_train, y_test = ydata[train_index], ydata[test_index]
    print("train/test sets stratified and split")
    return x_train, x_test, y_train, y_test


#random sample of training data
def reduce_dimensions(Xdata, ydata):
    nsamples, nx, ny = Xdata.shape
    Xdata2d = Xdata.reshape((nsamples,nx*ny))
    X_train, X_test, y_train, y_test = train_test_split(
        Xdata2d, ydata, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test


def reduce_dim_strat_over(Xdata,ydata):
    """
    Reduces the dimensions of inout data, splits into train/test set,
    then randomly oversamples the minority classes to match the majority class"""
    nsamples, nx, ny = Xdata.shape
    Xdata2d = Xdata.reshape((nsamples,nx*ny))
    stshsp = StratifiedShuffleSplit(n_splits= 1, test_size =0.2, random_state=42)
    train_index, test_index = next(stshsp.split(Xdata2d,ydata))
    x_train, x_test = Xdata2d[train_index], Xdata2d[test_index]
    y_train, y_test = ydata[train_index], ydata[test_index]
    ros = RandomOverSampler(random_state=0)
    X_resampled, y_resampled = ros.fit_resample(x_train, y_train)
    print("train/test sets stratified and split")
    return X_resampled, x_test, y_resampled, y_test



def reduce_dim_sampler(Xdata,ydata, sample_flag = False):
    """
    Reduces the dimensions of inout data, splits into train/test set,
    then randomly oversamples the minority classes to match the majority class"""
    nsamples, nx, ny = Xdata.shape
    Xdata2d = Xdata.reshape((nsamples,nx*ny))
    stshsp = StratifiedShuffleSplit(n_splits= 1, test_size =0.2, random_state=42)
    train_index, test_index = next(stshsp.split(Xdata2d,ydata))
    x_train, x_test = Xdata2d[train_index], Xdata2d[test_index]
    y_train, y_test = ydata[train_index], ydata[test_index]
    if sample_flag == 'o':
        ros = RandomOverSampler(random_state=0)
        X_resampled, y_resampled = ros.fit_resample(x_train, y_train)
    elif sample_flag == 's':
        X_resampled, y_resampled = SMOTE().fit_resample(x_train, y_train)
    elif sample_flag == 'a':
        X_resampled, y_resampled = ADASYN(n_neighbors=2).fit_resample(x_train, y_train)
    else:
        X_resampled, y_resampled = x_train, y_train
    print("train/test sets stratified and split")
    return X_resampled, x_test, y_resampled, y_test




def main():
    pull_window(df, window_size)
    slide_window(df, window_size)
    leaping_window(df, window_size)
    singlelabel_xy(windows, classdict)
    multilabel_xy(windows, classdict)
    reduce_dim_strat(Xdata,ydata)
    reduce_dimensions(Xdata,ydata)

if __name__ == "__main__":
    main()