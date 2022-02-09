from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix
#from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import classification_report
import pandas as pd
import numpy as np
#Input requires the arrays of xyz (Xtrain/xtest)
#  and the single number class(ytrain/ytest)

#TODO: have inputs of the parameters/hyperparameters and record those in the output


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
        
    return np.stack(Xdata), np.asarray(ydata)


def svm(X_train, X_test, y_train, y_test):
    parameters = {"degrees":3, "C":5, "kernel":"poly"}
    svm_clf = Pipeline([
            ("scalar", StandardScaler()),
            ("linear_svc", 
            SVC(kernel=parameters["kernel"],
                degree=parameters["degrees"],
                C=parameters["C"])),
    ])
    svm_clf.fit(X_train, y_train)
    ypred = svm_clf.predict(X_test)
    report = classification_report(
        y_test,
        ypred, 
        output_dict=True
        )
    parameters = svm_clf.get_params()
    # return report, parameters (dict with parameter names mapped to values)
    return report, parameters


def main():
    df = pd.read_csv("~/CNNworkspace/raterdata/dec21_cleanPennf1.csv")
    windows, classes = pull_window(df, 25)
    Xdata,ydata = construct_xy(windows)
    nsamples, nx, ny = Xdata.shape
    Xdata2d = Xdata.reshape((nsamples,nx*ny))
    X_train, X_test, y_train, y_test = train_test_split(
        Xdata2d, ydata, test_size=0.2, random_state=42)
    report = svm(X_train, X_test, y_train, y_test, classes)
    # svm_clf = Pipeline([
    #         ("scalar", StandardScaler()),
    #         ("linear_svc", SVC(kernel="poly",degree=3,C=5)),
    # ])
    # svm_clf.fit(X_train, y_train)
    # ypred = svm_clf.predict(X_test)
    # report = classification_report(
    #     y_test,
    #     ypred, 
    #     target_names=["s","l","c","a","d","i","w"],
    #     output_dict=True
    #     )
    reportdf = pd.DataFrame(report).transpose()
    reportdf.to_csv('svm_stats.csv')
    print(report)
    # scores = precision_recall_fscore_support(
    #     y_test, 
    #     ypred, 
    #     average=None,
    #     labels = ["s","l","t","c","a","d","i","w"]
    #     )
    # ytrainpred = cross_val_predict(svm_clf,X_train,y_train, cv=3)
    # conf_mx = confusion_matrix(y_train,ytrainpred,labels = [0,1,2,3,4,5,6])
    # #default uses the one vs one strategy, preferred as it is faster for a large
    # #training dataset

    

if __name__=="__main__":
    main()