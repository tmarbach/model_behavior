import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split




def pull_window(df, window_size):
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
        windows.append(window)
        classes.append(window.iloc[0]['behavior'])
    allclasses = set(classes)
    print("Windows pulled")
    return windows, list(allclasses)


def construct_xy(windows):
    """
    Input:
        windows -- list of dataframes of all one class 
    Output:
        Xdata -- arrays of xyz data of each window stacked together
        ydata -- integer class labels for each window
    """
    positions = ['acc_x', 'acc_y', 'acc_z']
    total_behaviors = ["s","l","t","c","a","d","i","w"]
    Xdata, ydata = [], []
    ### map each behavior to an integer ex: {'s': 0, 'l': 1, 't': 2, 'c': 3}
    mapping = {}
    for x in range(len(total_behaviors)):
        mapping[total_behaviors[x]] = x
    for window in windows:
        Xdata.append(window[positions].to_numpy())
        ydata.append(mapping[window['behavior'].iloc[0]])
        
    return np.stack(Xdata), np.asarray(ydata), mapping


def reduce_dimesions(Xdata, ydata):
    nsamples, nx, ny = Xdata.shape
    Xdata2d = Xdata.reshape((nsamples,nx*ny))
    X_train, X_test, y_train, y_test = train_test_split(
        Xdata2d, ydata, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test

# def main():


# if __name__=="__main__":
#     main()

