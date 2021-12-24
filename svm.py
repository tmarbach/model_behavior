from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix
import pandas as pd
import numpy as np
import csv
#Input requires the arrays of xyz (Xtrain/xtest)
#  and the single number class(ytrain/ytest)




def pull_window(df, window_size):
    """
    Input: 
    df -- dataframe of cleaned input data, likely from a csv
    window_size -- number of rows of data to convert to 1 row for AcceleRater (25 = 1sec)
    Output:
    windows -- list of lists of accel data (EX:[x,y,z,...,x,y,z,class_label])
    """
    if window_size > df.shape[0]:
        raise ValueError('Window larger than data given')
    windows = []
    number_of_rows_minus_window = df.shape[0] - window_size + 1
    for i in range(0, number_of_rows_minus_window, window_size):
        window = df[i:i+window_size]
        if len(set(window.behavior)) != 1:
            continue
        if len(set(np.ediff1d(window.input_index))) != 1:
             continue
        windows.append(window)
    return windows


def construct_xy(windows):
    """
    Input:
        windows -- list of dataframes of all one class 
    Output:
        total_data -- list of lists of flattened accel datawith class label a the end
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
        
    return np.stack(Xdata), ydata


def main():
    df = pd.read_csv("~/CNNworkspace/raterdata/dec21_cleanPennf1.csv")
    windows = pull_window(df, 5)
    Xdata,ydata = construct_xy(windows)
   # with open("CNNworkspace/raterdata/dec21Pennf1flat_data.csv", "w", newline="") as f:
  #      writer = csv.writer(f)
    #    writer.writerows(all_data)


    X_train, X_test, y_train, y_test = train_test_split(
...     Xdata, ydata, test_size=0.2, random_state=42)
    svm_clf = SVC()
    #default uses the one vs one strategy, preferred as it is faster for a large
    #training dataset
    svm_clf.fit(X_train, y_train)
    y_pred = svm_clf.predict(X_test)

    

if __name__=="__main__":
    main()